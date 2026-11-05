"""Keyword, vector and hybrid retrieval over chunks, collapsed to one result per document."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .db import to_vector_literal
from .fusion import reciprocal_rank_fusion

if TYPE_CHECKING:
    from psycopg_pool import AsyncConnectionPool

    from .embeddings import Embedder

SearchMode = Literal["keyword", "vector", "hybrid"]
SEARCH_MODES: tuple[SearchMode, ...] = ("keyword", "vector", "hybrid")

# plainto_tsquery joins terms with AND, which rarely matches a full natural-language query.
# Rewriting the operators to OR makes every term optional and lets ts_rank_cd reward
# chunks that contain more of them, closer together. See docs/adr/0002.
_KEYWORD_SQL = """
with q as (
    select replace(plainto_tsquery('english', %(query)s)::text, ' & ', ' | ')::tsquery as query
),
best as (
    select distinct on (c.document_id)
        c.document_id, c.id, ts_rank_cd(c.search_vector, q.query) as score
    from chunks c, q
    where c.search_vector @@ q.query
    order by c.document_id, score desc, c.id
)
select document_id, id, score from best order by score desc, document_id limit %(limit)s
"""

# The inner query must order by the distance operator alone for the HNSW index to be used.
_VECTOR_SQL = """
with nearest as materialized (
    select id, document_id, embedding <=> %(vector)s::vector as distance
    from chunks
    order by embedding <=> %(vector)s::vector
    limit %(chunks)s
),
best as (
    select distinct on (document_id) document_id, id, distance
    from nearest
    order by document_id, distance, id
)
select document_id, id, 1 - distance as score from best order by distance, document_id limit %(limit)s
"""


@dataclass(frozen=True, slots=True)
class Candidate:
    document_id: str
    chunk_id: int
    score: float


@dataclass(frozen=True, slots=True)
class RankedDocument:
    document_id: str
    chunk_id: int
    score: float
    keyword_rank: int | None
    vector_rank: int | None


@dataclass(frozen=True, slots=True)
class SearchHit:
    document_id: str
    title: str
    snippet: str
    score: float
    keyword_rank: int | None
    vector_rank: int | None


class SearchService:
    def __init__(
        self,
        pool: AsyncConnectionPool,
        embedder: Embedder,
        candidates: int = 100,
        ef_search: int = 100,
        rrf_k: int = 60,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        self._pool = pool
        self._embedder = embedder
        self.candidates = candidates
        self.ef_search = ef_search
        self.rrf_k = rrf_k
        self.weights = dict(weights or {"keyword": 1.0, "vector": 1.0})

    async def keyword(self, query: str, limit: int) -> list[Candidate]:
        async with self._pool.connection() as conn:
            cursor = await conn.execute(_KEYWORD_SQL, {"query": query, "limit": limit})
            rows = await cursor.fetchall()
        return [Candidate(document_id, chunk_id, float(score)) for document_id, chunk_id, score in rows]

    async def vector(
        self,
        vector: Sequence[float],
        limit: int,
        ef_search: int | None = None,
        exact: bool = False,
    ) -> list[Candidate]:
        """Nearest documents by cosine similarity of their best chunk.

        With exact=True the index is bypassed and every chunk is compared, which gives the
        ground truth for measuring HNSW recall.
        """
        params = {"vector": to_vector_literal(vector), "chunks": limit * 2, "limit": limit}
        async with self._pool.connection() as conn, conn.transaction():
            if exact:
                await conn.execute("select set_config('enable_indexscan', 'off', true)")
            else:
                await conn.execute(
                    "select set_config('hnsw.ef_search', %s, true)", (str(ef_search or self.ef_search),)
                )
                # Keeps scanning the graph when ef_search is smaller than the requested row count,
                # while still returning rows in exact distance order (pgvector 0.8 and later).
                await conn.execute("select set_config('hnsw.iterative_scan', 'strict_order', true)")
            cursor = await conn.execute(_VECTOR_SQL, params)
            rows = await cursor.fetchall()
        return [Candidate(document_id, chunk_id, float(score)) for document_id, chunk_id, score in rows]

    async def rank(self, query: str, mode: SearchMode, limit: int) -> list[RankedDocument]:
        if mode == "keyword":
            return _single(await self.keyword(query, limit), "keyword")
        if mode == "vector":
            return _single(await self.vector(await self._embedder.embed_query(query), limit), "vector")

        keyword, vector = await asyncio.gather(
            self.keyword(query, self.candidates),
            self._vector_for_text(query, self.candidates),
        )
        return fuse(keyword, vector, limit, self.rrf_k, self.weights)

    async def search(self, query: str, mode: SearchMode, limit: int) -> list[SearchHit]:
        ranked = await self.rank(query, mode, limit)
        if not ranked:
            return []
        async with self._pool.connection() as conn:
            cursor = await conn.execute(
                """
                select c.id, d.title, c.content
                from chunks c join documents d on d.id = c.document_id
                where c.id = any(%s)
                """,
                ([doc.chunk_id for doc in ranked],),
            )
            details = {chunk_id: (title, content) for chunk_id, title, content in await cursor.fetchall()}

        hits = []
        for doc in ranked:
            # A document deleted between ranking and this lookup is skipped.
            if doc.chunk_id not in details:
                continue
            title, content = details[doc.chunk_id]
            hits.append(
                SearchHit(
                    doc.document_id, title, snippet(content), doc.score, doc.keyword_rank, doc.vector_rank
                )
            )
        return hits

    async def _vector_for_text(self, query: str, limit: int) -> list[Candidate]:
        return await self.vector(await self._embedder.embed_query(query), limit)


def fuse(
    keyword: Sequence[Candidate],
    vector: Sequence[Candidate],
    limit: int,
    rrf_k: int = 60,
    weights: Mapping[str, float] | None = None,
) -> list[RankedDocument]:
    """Combines two document rankings with RRF.

    The chunk shown for a fused document is the one the vector retriever matched, falling
    back to the keyword match.
    """
    keyword_rank = {c.document_id: rank for rank, c in enumerate(keyword, start=1)}
    vector_rank = {c.document_id: rank for rank, c in enumerate(vector, start=1)}
    chunk = {c.document_id: c.chunk_id for c in keyword} | {c.document_id: c.chunk_id for c in vector}
    fused = reciprocal_rank_fusion(
        {"keyword": [c.document_id for c in keyword], "vector": [c.document_id for c in vector]},
        weights=weights,
        k=rrf_k,
    )
    return [
        RankedDocument(
            document_id,
            chunk[document_id],
            score,
            keyword_rank.get(document_id),
            vector_rank.get(document_id),
        )
        for document_id, score in fused[:limit]
    ]


def snippet(text: str, max_chars: int = 280) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    cut = text.rfind(" ", 0, max_chars)
    return text[: cut if cut > 0 else max_chars] + "..."


def _single(candidates: Sequence[Candidate], source: Literal["keyword", "vector"]) -> list[RankedDocument]:
    return [
        RankedDocument(
            c.document_id,
            c.chunk_id,
            c.score,
            keyword_rank=rank if source == "keyword" else None,
            vector_rank=rank if source == "vector" else None,
        )
        for rank, c in enumerate(candidates, start=1)
    ]
