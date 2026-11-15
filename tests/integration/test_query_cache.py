"""The shared query embedding cache against a real PostgreSQL database."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from psycopg_pool import AsyncConnectionPool

from hybrid_search.query_cache import PostgresQueryCache

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


@pytest.fixture
async def cache_pool(pool: AsyncConnectionPool) -> AsyncIterator[AsyncConnectionPool]:
    async with pool.connection() as conn:
        await conn.execute("truncate query_embedding_cache")
    yield pool


def cache(pool: AsyncConnectionPool, **overrides: object) -> PostgresQueryCache:
    options: dict[str, object] = {"ttl_seconds": 3600.0, "max_rows": 1000, "prune_every": 1000}
    options.update(overrides)
    return PostgresQueryCache(pool, "model-a", 3, **options)  # type: ignore[arg-type]


async def rows(pool: AsyncConnectionPool) -> int:
    async with pool.connection() as conn:
        cursor = await conn.execute("select count(*) from query_embedding_cache")
        row = await cursor.fetchone()
    assert row is not None
    return int(row[0])


async def test_put_then_get_from_another_instance(cache_pool: AsyncConnectionPool) -> None:
    writer, reader = cache(cache_pool), cache(cache_pool)
    assert await reader.get("q: aspirin") is None
    await writer.put("q: aspirin", [0.25, -1.5, 3.0])
    assert await reader.get("q: aspirin") == pytest.approx([0.25, -1.5, 3.0])
    assert (reader.hits, reader.misses) == (1, 1)


async def test_entries_are_separate_per_model_and_text(cache_pool: AsyncConnectionPool) -> None:
    await cache(cache_pool).put("q: aspirin", [1.0, 2.0, 3.0])
    other_model = PostgresQueryCache(cache_pool, "model-b", 3, 3600.0, 1000)
    assert await other_model.get("q: aspirin") is None
    assert await cache(cache_pool).get("q: Aspirin") is None


async def test_expired_entries_are_misses_and_are_replaced(cache_pool: AsyncConnectionPool) -> None:
    store = cache(cache_pool)
    await store.put("q: old", [1.0, 1.0, 1.0])
    async with cache_pool.connection() as conn:
        await conn.execute("update query_embedding_cache set created_at = now() - interval '2 hours'")
    assert await store.get("q: old") is None
    await store.put("q: old", [2.0, 2.0, 2.0])
    assert await store.get("q: old") == pytest.approx([2.0, 2.0, 2.0])


async def test_a_fresh_entry_is_not_overwritten(cache_pool: AsyncConnectionPool) -> None:
    store = cache(cache_pool)
    await store.put("q: x", [1.0, 1.0, 1.0])
    await store.put("q: x", [5.0, 5.0, 5.0])
    assert await store.get("q: x") == pytest.approx([1.0, 1.0, 1.0])


async def test_pruning_removes_expired_then_oldest_rows(cache_pool: AsyncConnectionPool) -> None:
    store = cache(cache_pool, max_rows=5, prune_every=1000)
    for i in range(8):
        await store.put(f"q: {i}", [float(i), 0.0, 0.0])
    async with cache_pool.connection() as conn:
        # Two rows are expired; the other six get increasing ages so the oldest are known.
        await conn.execute(
            "update query_embedding_cache set created_at = now() - interval '2 hours' "
            "where embedding[1] in (0, 1)"
        )
        for i in range(2, 8):
            await conn.execute(
                "update query_embedding_cache set created_at = now() - make_interval(secs => %s) "
                "where embedding[1] = %s",
                (100 - i, i),
            )
    pruner = cache(cache_pool, max_rows=5, prune_every=1)
    await pruner.put("q: newest", [9.0, 0.0, 0.0])
    # 9 rows: 2 expired are removed, then the oldest 2 of the remaining 7.
    assert await rows(cache_pool) == 5
    for i in (0, 1, 2, 3):
        assert await pruner.get(f"q: {i}") is None
    for text in ("q: 4", "q: 7", "q: newest"):
        assert await pruner.get(text) is not None


async def test_wrong_dimensions_are_misses(cache_pool: AsyncConnectionPool) -> None:
    await PostgresQueryCache(cache_pool, "model-a", 2, 3600.0, 1000).put("q: x", [1.0, 2.0])
    assert await cache(cache_pool).get("q: x") is None


async def test_database_errors_are_misses(cache_pool: AsyncConnectionPool) -> None:
    store = cache(cache_pool)
    async with cache_pool.connection() as conn:
        await conn.execute("alter table query_embedding_cache rename to query_embedding_cache_away")
    try:
        assert await store.get("q: x") is None
        await store.put("q: x", [1.0, 2.0, 3.0])  # does not raise
        assert store.errors == 2
    finally:
        async with cache_pool.connection() as conn:
            await conn.execute("alter table query_embedding_cache_away rename to query_embedding_cache")
