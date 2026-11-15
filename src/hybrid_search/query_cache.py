"""Query embedding cache shared across API processes, stored in PostgreSQL (docs/adr/0009).

Each API process keeps its own in-memory cache in front of this one. On a miss there, the
process looks the query up here before calling the embedding provider, so a query embedded
by one process is reused by the others. The cache is an optimisation only: any database
error is logged and treated as a miss, and searches carry on with the provider.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import TYPE_CHECKING

import psycopg

if TYPE_CHECKING:
    from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger(__name__)

_GET_SQL = """
select embedding from query_embedding_cache
where model = %(model)s and query_hash = %(hash)s
  and created_at > now() - make_interval(secs => %(ttl)s)
"""

_PUT_SQL = """
insert into query_embedding_cache (model, query_hash, embedding)
values (%(model)s, %(hash)s, %(embedding)s)
on conflict (model, query_hash) do update
    set embedding = excluded.embedding, created_at = excluded.created_at
    where query_embedding_cache.created_at <= now() - make_interval(secs => %(ttl)s)
"""

# Pruning runs as two statements: in one statement both deletes would see the same
# snapshot, so the row count would still include the expired rows being deleted.
_PRUNE_EXPIRED_SQL = """
delete from query_embedding_cache where created_at <= now() - make_interval(secs => %(ttl)s)
"""

_PRUNE_OLDEST_SQL = """
delete from query_embedding_cache
where ctid in (
    select ctid from query_embedding_cache
    order by created_at
    limit greatest((select count(*) from query_embedding_cache) - %(max_rows)s, 0)
)
"""

# Minimum time between two warnings about the cache being unavailable.
_WARNING_INTERVAL_SECONDS = 60.0


def query_key(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


class PostgresQueryCache:
    """Stores query embeddings by model and SHA-256 of the text sent to the provider.

    Entries expire ttl_seconds after they were written. Every prune_every writes, expired
    entries and the oldest entries beyond max_rows are deleted, so the table stays bounded
    without a background task.
    """

    def __init__(
        self,
        pool: AsyncConnectionPool,
        model: str,
        dimensions: int,
        ttl_seconds: float,
        max_rows: int,
        prune_every: int = 1000,
    ) -> None:
        self._pool = pool
        self._model = model
        self._dimensions = dimensions
        self._ttl = ttl_seconds
        self._max_rows = max_rows
        self._prune_every = prune_every
        self._writes_since_prune = 0
        self._last_warning = float("-inf")
        self.hits = 0
        self.misses = 0
        self.errors = 0

    async def get(self, text: str) -> list[float] | None:
        try:
            async with self._pool.connection() as conn:
                cursor = await conn.execute(
                    _GET_SQL, {"model": self._model, "hash": query_key(text), "ttl": self._ttl}
                )
                row = await cursor.fetchone()
        except psycopg.Error as error:  # includes pool timeouts and a closed pool
            self._failed("read", error)
            return None
        if row is None or len(row[0]) != self._dimensions:
            self.misses += 1
            return None
        self.hits += 1
        return [float(x) for x in row[0]]

    async def put(self, text: str, embedding: list[float]) -> None:
        params = {"model": self._model, "hash": query_key(text), "embedding": embedding, "ttl": self._ttl}
        try:
            async with self._pool.connection() as conn:
                await conn.execute(_PUT_SQL, params)
                self._writes_since_prune += 1
                if self._writes_since_prune >= self._prune_every:
                    self._writes_since_prune = 0
                    await conn.execute(_PRUNE_EXPIRED_SQL, {"ttl": self._ttl})
                    await conn.execute(_PRUNE_OLDEST_SQL, {"max_rows": self._max_rows})
        except psycopg.Error as error:
            self._failed("write", error)

    def _failed(self, operation: str, error: Exception) -> None:
        self.errors += 1
        now = time.monotonic()
        if now - self._last_warning >= _WARNING_INTERVAL_SECONDS:
            self._last_warning = now
            logger.warning(
                "shared query embedding cache %s failed; using the embedding provider",
                operation,
                extra={"error": f"{type(error).__name__}: {error}"},
            )
