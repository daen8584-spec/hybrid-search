import asyncio
import math

import pytest
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from hybrid_search.db import embedding_dimensions, migrate
from hybrid_search.ingest import Ingestor
from hybrid_search.search import SearchService

from .conftest import DIMENSIONS, HashingEmbedder, TermOverlapReranker

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

ASPIRIN = (
    "Aspirin reduces fever and inflammation.\n\nLow dose aspirin lowers the risk of a second heart attack."
)
VITAMIN = "Vitamin D supplementation improves bone density in older adults."
SLEEP = "Sleep deprivation impairs memory consolidation and attention."


async def chunk_rows(pool: AsyncConnectionPool, document_id: str) -> list[tuple[int, str]]:
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "select ordinal, content from chunks where document_id = %s order by ordinal", (document_id,)
        )
        return [(ordinal, content) for ordinal, content in await cursor.fetchall()]


async def test_migrations_are_idempotent_and_match_the_embedding_size(database_url: str) -> None:
    async with await AsyncConnection.connect(database_url, autocommit=True) as conn:
        assert await migrate(conn) == []
        assert await embedding_dimensions(conn) == DIMENSIONS


async def test_upsert_creates_then_skips_unchanged_documents(
    pool: AsyncConnectionPool, embedder: HashingEmbedder
) -> None:
    ingestor = Ingestor(pool, embedder)

    created = await ingestor.upsert("d1", "Aspirin", ASPIRIN, {"source": "test"})
    unchanged = await ingestor.upsert("d1", "Aspirin", ASPIRIN, {"source": "test"})

    assert (created.status, created.chunks, created.embedded) == ("created", 1, 1)
    assert (unchanged.status, unchanged.chunks, unchanged.embedded) == ("unchanged", 1, 0)
    assert embedder.document_calls == 1


async def test_updates_embed_only_the_chunks_that_changed(
    pool: AsyncConnectionPool, embedder: HashingEmbedder
) -> None:
    ingestor = Ingestor(pool, embedder, max_tokens=12, overlap_tokens=0)
    paragraphs = [
        "Aspirin reduces fever and inflammation in adults.",
        "Vitamin D supplementation improves bone density.",
        "Sleep deprivation impairs memory consolidation.",
    ]
    first = await ingestor.upsert("d1", "", "\n\n".join(paragraphs))
    paragraphs[2] = "Regular exercise improves sleep quality."
    second = await ingestor.upsert("d1", "", "\n\n".join(paragraphs))

    assert (first.status, first.chunks, first.embedded) == ("created", 3, 3)
    assert (second.status, second.chunks, second.embedded, second.reused) == ("updated", 3, 1, 2)
    assert embedder.embedded_texts[-1] == "Regular exercise improves sleep quality."
    assert [content for _, content in await chunk_rows(pool, "d1")][2] == paragraphs[2]


async def test_identical_text_in_another_document_reuses_the_embedding(
    pool: AsyncConnectionPool, embedder: HashingEmbedder
) -> None:
    ingestor = Ingestor(pool, embedder)
    await ingestor.upsert("d1", "Shared", VITAMIN)
    copy = await ingestor.upsert("d2", "Shared", VITAMIN)
    assert (copy.status, copy.embedded, copy.reused) == ("created", 0, 1)


async def test_concurrent_writes_to_one_document_leave_a_consistent_state(
    pool: AsyncConnectionPool, embedder: HashingEmbedder
) -> None:
    ingestor = Ingestor(pool, embedder, max_tokens=12, overlap_tokens=0)
    versions = [
        "\n\n".join(f"Version {v} paragraph {p} about topic {p}." for p in range(4)) for v in range(5)
    ]

    await asyncio.gather(*(ingestor.upsert("d1", "", text) for text in versions))

    rows = await chunk_rows(pool, "d1")
    assert [ordinal for ordinal, _ in rows] == [0, 1, 2, 3]
    assert len({content.split()[1] for _, content in rows}) == 1, "chunks come from a single version"


async def test_delete_removes_the_document_and_its_chunks(
    pool: AsyncConnectionPool, embedder: HashingEmbedder
) -> None:
    ingestor = Ingestor(pool, embedder)
    await ingestor.upsert("d1", "Aspirin", ASPIRIN)

    assert await ingestor.delete("d1") is True
    assert await ingestor.delete("d1") is False
    assert await chunk_rows(pool, "d1") == []


async def test_rejects_empty_input(pool: AsyncConnectionPool, embedder: HashingEmbedder) -> None:
    ingestor = Ingestor(pool, embedder)
    with pytest.raises(ValueError):
        await ingestor.upsert("d1", "", "   ")
    with pytest.raises(ValueError):
        await ingestor.upsert(" ", "", "text")


@pytest.fixture
async def corpus(pool: AsyncConnectionPool, embedder: HashingEmbedder) -> SearchService:
    ingestor = Ingestor(pool, embedder)
    await ingestor.upsert("aspirin", "Aspirin", ASPIRIN)
    await ingestor.upsert("vitamin", "Vitamin D", VITAMIN)
    await ingestor.upsert("sleep", "Sleep", SLEEP)
    return SearchService(pool, embedder, candidates=10)


async def test_keyword_search_matches_any_query_term(corpus: SearchService) -> None:
    results = await corpus.rank("can aspirin prevent a heart attack", "keyword", 10)
    assert [doc.document_id for doc in results] == ["aspirin"]
    assert results[0].keyword_rank == 1 and results[0].vector_rank is None


async def test_keyword_search_handles_queries_without_lexemes(corpus: SearchService) -> None:
    assert await corpus.rank("the and of", "keyword", 10) == []


async def test_vector_search_returns_one_result_per_document(corpus: SearchService) -> None:
    results = await corpus.rank("bone density vitamin", "vector", 10)
    assert results[0].document_id == "vitamin"
    assert len({doc.document_id for doc in results}) == len(results) == 3


async def test_exact_and_indexed_vector_search_agree(
    corpus: SearchService, embedder: HashingEmbedder
) -> None:
    vector = await embedder.embed_query("memory and attention")
    exact = await corpus.vector(vector, 3, exact=True)
    indexed = await corpus.vector(vector, 3, ef_search=40)
    assert [c.document_id for c in indexed] == [c.document_id for c in exact]


async def test_hybrid_search_combines_both_retrievers(corpus: SearchService) -> None:
    results = await corpus.rank("aspirin fever", "hybrid", 10)
    top = results[0]
    assert top.document_id == "aspirin"
    assert top.keyword_rank == 1 and top.vector_rank == 1
    assert {doc.document_id for doc in results} == {"aspirin", "vitamin", "sleep"}


async def test_search_returns_titles_and_snippets(corpus: SearchService) -> None:
    [hit] = await corpus.search("vitamin", "keyword", 1)
    assert (hit.document_id, hit.title) == ("vitamin", "Vitamin D")
    assert hit.snippet.startswith("Vitamin D supplementation")


async def test_keyword_ranking_weights_rare_terms_above_common_ones(
    pool: AsyncConnectionPool, embedder: HashingEmbedder
) -> None:
    ingestor = Ingestor(pool, embedder)
    await ingestor.upsert("common", "", "This study of patients reports a study of patients in a study.")
    await ingestor.upsert("rare", "", "Melatonin was given to patients in this study.")
    for index in range(4):
        await ingestor.upsert(f"filler-{index}", "", f"Another study of patients, number {index}.")
    service = SearchService(pool, embedder)

    results = await service.rank("melatonin study patients", "keyword", 10)

    assert results[0].document_id == "rare"
    assert len(results) == 6


async def test_keyword_index_follows_updates_and_deletes(
    pool: AsyncConnectionPool, embedder: HashingEmbedder
) -> None:
    ingestor = Ingestor(pool, embedder)
    await ingestor.upsert("d1", "", "Aspirin reduces fever.")
    service = SearchService(pool, embedder, stats_ttl_seconds=0)
    assert [d.document_id for d in await service.rank("aspirin", "keyword", 10)] == ["d1"]

    await ingestor.upsert("d1", "", "Ibuprofen reduces fever.")
    assert await service.rank("aspirin", "keyword", 10) == []
    assert [d.document_id for d in await service.rank("ibuprofen", "keyword", 10)] == ["d1"]

    await ingestor.delete("d1")
    assert await service.rank("ibuprofen", "keyword", 10) == []
    async with pool.connection() as conn:
        cursor = await conn.execute("select count(*) from chunk_terms")
        assert await cursor.fetchone() == (0,)


async def test_keyword_search_on_an_empty_corpus(
    pool: AsyncConnectionPool, embedder: HashingEmbedder
) -> None:
    assert await SearchService(pool, embedder).rank("anything", "keyword", 10) == []


async def test_reranking_scores_the_top_candidates_by_title_and_text(
    pool: AsyncConnectionPool, embedder: HashingEmbedder
) -> None:
    texts = {"aspirin": ("Aspirin", ASPIRIN), "vitamin": ("Vitamin D", VITAMIN), "sleep": ("Sleep", SLEEP)}
    ingestor = Ingestor(pool, embedder)
    for document_id, (title, text) in texts.items():
        await ingestor.upsert(document_id, title, text)
    reranker = TermOverlapReranker()
    service = SearchService(
        pool, embedder, candidates=10, reranker=reranker, rerank_depth=2, rerank_weight=math.inf
    )
    query = "aspirin and bone density in older adults"

    plain = await service.rank(query, "hybrid", 3)
    reranked = await service.rank(query, "hybrid", 3, rerank=True)

    # Only the top two first-stage results are sent, each as its title and chunk text.
    [(sent_query, documents)] = reranker.calls
    assert sent_query == query
    expected = [f"{texts[d.document_id][0]}\n\n{texts[d.document_id][1]}" for d in plain[:2]]
    assert sorted(documents) == sorted(expected)
    # The vitamin abstract shares the most query words, so the reranker puts it first,
    # and the document beyond the rerank depth keeps its place without a rerank rank.
    assert reranked[0].document_id == "vitamin"
    assert reranked[0].rerank_rank == 1
    assert reranked[2].document_id == plain[2].document_id
    assert reranked[2].rerank_rank is None


async def test_reranking_requires_a_reranker(corpus: SearchService) -> None:
    with pytest.raises(ValueError, match="not configured"):
        await corpus.rank("aspirin", "hybrid", 3, rerank=True)
