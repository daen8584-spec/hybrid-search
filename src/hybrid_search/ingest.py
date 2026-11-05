"""Document ingestion: chunking, embedding with reuse, and an atomic replace of a document's chunks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from .chunking import chunk_text
from .db import to_vector_literal
from .embeddings import Embedder

IngestStatus = Literal["created", "updated", "unchanged"]


@dataclass(frozen=True, slots=True)
class IngestResult:
    document_id: str
    status: IngestStatus
    chunks: int
    embedded: int
    reused: int


def embedding_input(title: str, text: str) -> str:
    """The exact text sent to the embedding model for one chunk. The title gives every chunk context."""
    return f"{title}\n\n{text}" if title else text


def embedding_key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()


def document_hash(
    model: str, max_tokens: int, overlap_tokens: int, title: str, text: str, metadata: Mapping[str, Any]
) -> str:
    """Changes whenever the stored chunks would change, including a new model or chunking settings."""
    payload = json.dumps(
        [model, max_tokens, overlap_tokens, title, text, metadata], sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class Ingestor:
    def __init__(
        self,
        pool: AsyncConnectionPool,
        embedder: Embedder,
        max_tokens: int = 384,
        overlap_tokens: int = 48,
    ) -> None:
        self._pool = pool
        self._embedder = embedder
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens

    async def upsert(
        self, document_id: str, title: str, text: str, metadata: Mapping[str, Any] | None = None
    ) -> IngestResult:
        """Creates or replaces a document.

        Embeddings are computed before the write transaction starts, so a slow embedding call
        never holds locks. Chunks whose embedding input is unchanged reuse the stored vector.
        """
        if not document_id.strip():
            raise ValueError("document id must not be empty")
        if not text.strip():
            raise ValueError("document text must not be empty")
        metadata = dict(metadata or {})
        model = self._embedder.model
        content_hash = document_hash(model, self._max_tokens, self._overlap_tokens, title, text, metadata)

        async with self._pool.connection() as conn:
            cursor = await conn.execute("select content_hash from documents where id = %s", (document_id,))
            row = await cursor.fetchone()
        exists = row is not None
        if row is not None and row[0] == content_hash:
            count = await self._chunk_count(document_id)
            return IngestResult(document_id, "unchanged", count, 0, count)

        chunks = chunk_text(text, self._max_tokens, self._overlap_tokens)
        inputs = [embedding_input(title, chunk.text) for chunk in chunks]
        keys = [embedding_key(model, value) for value in inputs]

        stored = await self._stored_embeddings(sorted(set(keys)))
        missing: dict[str, str] = {}
        for key, value in zip(keys, inputs, strict=True):
            if key not in stored:
                missing.setdefault(key, value)
        if missing:
            vectors = await self._embedder.embed_documents(list(missing.values()))
            stored.update(
                (key, to_vector_literal(vector)) for key, vector in zip(missing, vectors, strict=True)
            )

        async with self._pool.connection() as conn, conn.transaction():
            # Serializes concurrent writes to the same document; different documents proceed in parallel.
            await conn.execute("select pg_advisory_xact_lock(hashtext(%s))", (document_id,))
            await conn.execute(
                """
                insert into documents (id, title, content_hash, metadata)
                values (%s, %s, %s, %s)
                on conflict (id) do update
                set title = excluded.title, content_hash = excluded.content_hash,
                    metadata = excluded.metadata, updated_at = now()
                """,
                (document_id, title, content_hash, Jsonb(metadata)),
            )
            await conn.execute("delete from chunks where document_id = %s", (document_id,))
            async with conn.cursor() as cursor:
                await cursor.executemany(
                    """
                    insert into chunks
                        (document_id, ordinal, heading, content, token_count, embedding_key, embedding)
                    values (%s, %s, %s, %s, %s, %s, %s::vector)
                    """,
                    [
                        (document_id, chunk.ordinal, title, chunk.text, chunk.token_count, key, stored[key])
                        for chunk, key in zip(chunks, keys, strict=True)
                    ],
                )
            # Postings for BM25. Old ones were removed with the old chunks by the foreign key cascade.
            await conn.execute(
                """
                insert into chunk_terms (lexeme, chunk_id, document_id, tf, chunk_length)
                select t.lexeme, c.id, c.document_id, greatest(cardinality(t.positions), 1), c.term_count
                from chunks c cross join lateral unnest(c.search_vector) as t
                where c.document_id = %s
                """,
                (document_id,),
            )

        return IngestResult(
            document_id=document_id,
            status="updated" if exists else "created",
            chunks=len(chunks),
            embedded=len(missing),
            reused=len(chunks) - sum(1 for key in keys if key in missing),
        )

    async def delete(self, document_id: str) -> bool:
        async with self._pool.connection() as conn:
            cursor = await conn.execute("delete from documents where id = %s", (document_id,))
            return cursor.rowcount > 0

    async def _chunk_count(self, document_id: str) -> int:
        async with self._pool.connection() as conn:
            cursor = await conn.execute("select count(*) from chunks where document_id = %s", (document_id,))
            row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def _stored_embeddings(self, keys: list[str]) -> dict[str, str]:
        if not keys:
            return {}
        async with self._pool.connection() as conn:
            cursor = await conn.execute(
                """
                select distinct on (embedding_key) embedding_key, embedding::text
                from chunks where embedding_key = any(%s)
                """,
                (keys,),
            )
            return {key: vector for key, vector in await cursor.fetchall()}
