"""Keyword, vector and hybrid retrieval over chunks, collapsed to one result per document."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from .db import to_vector_literal
from .fusion import reciprocal_rank_fusion

if TYPE_CHECKING:
    from psycopg_pool import AsyncConnectionPool

    from .config import Settings
    from .embeddings import Embedder
    from .reranker import Reranker

SearchMode = Literal["keyword", "vector", "hybrid"]
SEARCH_MODES: tuple[SearchMode, ...] = ("keyword", "vector", "hybrid")

# BM25 over the chunk_terms inverted index (docs/adr/0007).
#
# Candidates are the chunks that contain at least one selective query term: a term whose
# document frequency is at most max_df, or one of the rarest terms until their combined
# postings reach min_candidates. Candidates are then scored with every query term. Chunks
# that contain only very common terms are skipped, which bounds the work per query without
# changing the top results in practice, because those terms carry little weight.
_KEYWORD_SQL = """
with query_terms as (
    select distinct t.lexeme from unnest(to_tsvector('english', %(query)s)) as t
),
term_weights as (
    select q.lexeme, df.df, ln(1 + (%(n)s - df.df + 0.5) / (df.df + 0.5)) as idf
    from query_terms q
    cross join lateral (select count(*) as df from chunk_terms p where p.lexeme = q.lexeme) df
    where df.df > 0
),
ordered as (
    select lexeme, df,
        coalesce(sum(df) over (order by df, lexeme rows between unbounded preceding and 1 preceding), 0)
            as rarer_df
    from term_weights
),
selective as (
    select lexeme from ordered where df <= %(max_df)s or rarer_df < %(min_candidates)s
),
candidates as (
    select distinct p.chunk_id from selective s join chunk_terms p on p.lexeme = s.lexeme
),
scored as (
    select p.document_id, p.chunk_id,
        sum(
            w.idf * p.tf * (%(k1)s + 1)
            / (p.tf + %(k1)s * (1 - %(b)s + %(b)s * p.chunk_length / %(avgdl)s))
        ) as score
    from candidates c
    cross join term_weights w
    join chunk_terms p on p.lexeme = w.lexeme and p.chunk_id = c.chunk_id
    group by p.document_id, p.chunk_id
),
best as (
    select distinct on (document_id) document_id, chunk_id, score
    from scored
    order by document_id, score desc, chunk_id
)
select document_id, chunk_id, score from best order by score desc, document_id limit %(limit)s
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
select document_id, id, 1 - distance as score, (select count(*) from nearest) as fetched
from best order by distance, document_id limit %(limit)s
"""

# Vector search ranks chunks, and a document can own several of the nearest chunks, so the
# first fetch takes twice as many chunks as documents requested. When those chunks cover
# fewer than `limit` documents, the fetch is repeated with four times as many chunks, up to
# this multiple of `limit`. A document is missed only when more chunks than that rank above
# its best chunk.
_VECTOR_MAX_CHUNKS_PER_RESULT = 32


@dataclass(frozen=True, slots=True)
class CorpusStats:
    chunks: int
    mean_length: float


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
    rerank_rank: int | None = None


@dataclass(frozen=True, slots=True)
class SearchHit:
    document_id: str
    title: str
    snippet: str
    score: float
    keyword_rank: int | None
    vector_rank: int | None
    rerank_rank: int | None = None


class SearchService:
    def __init__(
        self,
        pool: AsyncConnectionPool,
        embedder: Embedder,
        candidates: int = 100,
        ef_search: int = 40,
        rrf_k: int = 60,
        weights: Mapping[str, float] | None = None,
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
        max_df_fraction: float = 0.1,
        min_candidates: int = 1_000,
        stats_ttl_seconds: float = 60.0,
        reranker: Reranker | None = None,
        rerank_depth: int = 20,
        rerank_weight: float = 0.25,
    ) -> None:
        self._pool = pool
        self._embedder = embedder
        self.candidates = candidates
        self.ef_search = ef_search
        self.rrf_k = rrf_k
        self.weights = dict(weights or {"keyword": 1.0, "vector": 1.0})
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.max_df_fraction = max_df_fraction
        self.min_candidates = min_candidates
        self._stats_ttl = stats_ttl_seconds
        self._stats: tuple[float, CorpusStats] | None = None
        self.reranker = reranker
        self.rerank_depth = rerank_depth
        self.rerank_weight = rerank_weight

    @classmethod
    def from_settings(
        cls,
        pool: AsyncConnectionPool,
        embedder: Embedder,
        settings: Settings,
        reranker: Reranker | None = None,
    ) -> SearchService:
        return cls(
            pool,
            embedder,
            ef_search=settings.hnsw_ef_search,
            weights={"keyword": settings.fusion_keyword_weight, "vector": 1.0},
            max_df_fraction=settings.bm25_max_df_fraction,
            reranker=reranker,
            rerank_depth=settings.rerank_depth,
            rerank_weight=settings.rerank_weight,
        )

    async def corpus_stats(self) -> CorpusStats:
        """Chunk count and mean chunk length for BM25, cached for stats_ttl_seconds.

        These change slowly as documents are added, so a short-lived cache avoids a
        full count on every query at the cost of slightly stale length normalization.
        """
        now = time.monotonic()
        if self._stats is not None and now - self._stats[0] < self._stats_ttl:
            return self._stats[1]
        async with self._pool.connection() as conn:
            cursor = await conn.execute("select count(*), coalesce(avg(term_count), 0) from chunks")
            row = await cursor.fetchone()
        stats = CorpusStats(chunks=int(row[0]), mean_length=float(row[1])) if row else CorpusStats(0, 0.0)
        self._stats = (now, stats)
        return stats

    async def keyword(self, query: str, limit: int) -> list[Candidate]:
        stats = await self.corpus_stats()
        if stats.chunks == 0:
            return []
        params = {
            "query": query,
            "limit": limit,
            "n": stats.chunks,
            "avgdl": max(stats.mean_length, 1.0),
            "k1": self.bm25_k1,
            "b": self.bm25_b,
            "max_df": int(stats.chunks * self.max_df_fraction),
            "min_candidates": self.min_candidates,
        }
        async with self._pool.connection() as conn:
            cursor = await conn.execute(_KEYWORD_SQL, params)
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
        chunks = limit * 2
        while True:
            rows, fetched = await self._nearest(vector, chunks, limit, ef_search, exact)
            if len(rows) >= limit or fetched < chunks or chunks >= limit * _VECTOR_MAX_CHUNKS_PER_RESULT:
                return rows
            chunks = min(chunks * 4, limit * _VECTOR_MAX_CHUNKS_PER_RESULT)

    async def _nearest(
        self, vector: Sequence[float], chunks: int, limit: int, ef_search: int | None, exact: bool
    ) -> tuple[list[Candidate], int]:
        """Best chunk per document among the `chunks` nearest, and how many chunks were found."""
        params = {"vector": to_vector_literal(vector), "chunks": chunks, "limit": limit}
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
                # The planner's HNSW cost estimate grows with ef_search, and on a corpus of this size
                # it switches to a sequential scan from ef_search 80 upward although the index is
                # several times faster. Approximate mode always uses the index (docs/adr/0005).
                await conn.execute("select set_config('enable_seqscan', 'off', true)")
            cursor = await conn.execute(_VECTOR_SQL, params)
            rows = await cursor.fetchall()
        fetched = int(rows[0][3]) if rows else 0
        return [
            Candidate(document_id, chunk_id, float(score)) for document_id, chunk_id, score, _ in rows
        ], fetched

    async def rank(
        self, query: str, mode: SearchMode, limit: int, rerank: bool = False
    ) -> list[RankedDocument]:
        """Ranks documents for a query. With rerank=True, the top `rerank_depth` results of
        the chosen mode are rescored by the reranker and combined with their first-stage
        ranks (see apply_rerank)."""
        depth = max(limit, self.rerank_depth) if rerank else limit
        ranked = await self._first_stage(query, mode, depth)
        if not rerank:
            return ranked
        if self.reranker is None:
            raise ValueError("reranking is not configured")
        scores = await self.rerank_scores(query, ranked[: self.rerank_depth])
        return apply_rerank(ranked, scores, self.rerank_weight, self.rrf_k)[:limit]

    async def rerank_scores(self, query: str, ranked: Sequence[RankedDocument]) -> list[float]:
        """Reranker scores for the given documents, from the chunk each was ranked by."""
        assert self.reranker is not None
        texts = await self._chunk_texts([doc.chunk_id for doc in ranked])
        documents = []
        for doc in ranked:
            title, content = texts.get(doc.chunk_id, ("", ""))
            documents.append(f"{title}\n\n{content}" if title else content)
        return await self.reranker.score(query, documents)

    async def _first_stage(self, query: str, mode: SearchMode, limit: int) -> list[RankedDocument]:
        if mode == "keyword":
            return _single(await self.keyword(query, limit), "keyword")
        if mode == "vector":
            return _single(await self.vector(await self._embedder.embed_query(query), limit), "vector")

        keyword, vector = await asyncio.gather(
            self.keyword(query, self.candidates),
            self._vector_for_text(query, self.candidates),
        )
        return fuse(keyword, vector, limit, self.rrf_k, self.weights)

    async def _chunk_texts(self, chunk_ids: Sequence[int]) -> dict[int, tuple[str, str]]:
        if not chunk_ids:
            return {}
        async with self._pool.connection() as conn:
            cursor = await conn.execute(
                """
                select c.id, d.title, c.content
                from chunks c join documents d on d.id = c.document_id
                where c.id = any(%s)
                """,
                (list(chunk_ids),),
            )
            return {chunk_id: (title, content) for chunk_id, title, content in await cursor.fetchall()}

    async def search(self, query: str, mode: SearchMode, limit: int, rerank: bool = False) -> list[SearchHit]:
        ranked = await self.rank(query, mode, limit, rerank)
        details = await self._chunk_texts([doc.chunk_id for doc in ranked])
        hits = []
        for doc in ranked:
            # A document deleted between ranking and this lookup is skipped.
            if doc.chunk_id not in details:
                continue
            title, content = details[doc.chunk_id]
            hits.append(
                SearchHit(
                    doc.document_id,
                    title,
                    snippet(content),
                    doc.score,
                    doc.keyword_rank,
                    doc.vector_rank,
                    doc.rerank_rank,
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


def apply_rerank(
    ranked: Sequence[RankedDocument],
    scores: Sequence[float],
    weight: float,
    rrf_k: int = 60,
) -> list[RankedDocument]:
    """Combines a first-stage ranking with reranker scores for its first len(scores) documents.

    The reranker's order is fused with the first-stage order by RRF, with the reranker's
    ranking weighted by `weight` relative to the first stage. A weight of 0 keeps the
    first-stage order, and an infinite weight uses the reranker's order for the reranked
    documents, followed by the rest in their original order. Fusing instead of replacing
    keeps a weak reranker from discarding a strong first-stage ranking; the right weight
    is measured with `hybrid-search tune-rerank`.
    """
    depth = len(scores)
    head = list(ranked[:depth])
    order = sorted(range(depth), key=lambda i: scores[i], reverse=True)
    rerank_rank = {head[i].document_id: position for position, i in enumerate(order, start=1)}

    if math.isinf(weight):
        reordered = [head[i] for i in order] + list(ranked[depth:])
        fused_scores = {
            doc.document_id: float(len(reordered) - position) for position, doc in enumerate(reordered)
        }
    else:
        fused = reciprocal_rank_fusion(
            {
                "first_stage": [doc.document_id for doc in ranked],
                "rerank": [head[i].document_id for i in order],
            },
            weights={"first_stage": 1.0, "rerank": weight},
            k=rrf_k,
        )
        fused_scores = dict(fused)
        by_id = {doc.document_id: doc for doc in ranked}
        reordered = [by_id[document_id] for document_id, _ in fused]

    return [
        replace(doc, score=fused_scores[doc.document_id], rerank_rank=rerank_rank.get(doc.document_id))
        for doc in reordered
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
