"""Integration tests run against a real PostgreSQL database with pgvector.

The database comes from TEST_DATABASE_URL, which defaults to DATABASE_URL with the database
name replaced by hybrid_search_test. Every test starts from empty tables.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import re
from collections.abc import AsyncIterator, Sequence

import pytest
from psycopg import AsyncConnection
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg_pool import AsyncConnectionPool

from hybrid_search.config import load_dotenv
from hybrid_search.db import migrate, open_pool

DIMENSIONS = 768


class HashingEmbedder:
    """Deterministic embeddings from hashed word features: texts that share words are similar."""

    model = "hashing-test-embedder"
    dimensions = DIMENSIONS

    def __init__(self) -> None:
        self.document_calls = 0
        self.embedded_texts: list[str] = []

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_calls += 1
        self.embedded_texts.extend(texts)
        return [self._vector(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        vector = [0.0] * DIMENSIONS
        for word in re.findall(r"\w+", text.lower()):
            digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
            vector[int.from_bytes(digest[:4], "big") % DIMENSIONS] += 1.0 if digest[4] % 2 else -1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


class TermOverlapReranker:
    """Scores a document by how many distinct query words it contains, so tests can
    predict the reranked order."""

    model = "term-overlap-test-reranker"

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    async def score(self, query: str, documents: Sequence[str]) -> list[float]:
        self.calls.append((query, list(documents)))
        words = set(re.findall(r"\w+", query.lower()))
        return [float(len(words & set(re.findall(r"\w+", d.lower())))) for d in documents]


def resolve_test_database_url() -> str:
    load_dotenv()
    main = os.environ.get("DATABASE_URL", "postgresql:///hybrid_search")
    url = os.environ.get("TEST_DATABASE_URL") or make_conninfo(main, dbname="hybrid_search_test")
    if conninfo_to_dict(url).get("dbname") == conninfo_to_dict(main).get("dbname"):
        raise RuntimeError("the test database must differ from DATABASE_URL: the tests truncate every table")
    return url


@pytest.fixture(scope="session")
def database_url() -> str:
    url = resolve_test_database_url()

    async def prepare() -> None:
        async with await AsyncConnection.connect(url, autocommit=True) as conn:
            await migrate(conn)

    asyncio.run(prepare())
    return url


@pytest.fixture
async def pool(database_url: str) -> AsyncIterator[AsyncConnectionPool]:
    pool = await open_pool(database_url, max_size=6)
    async with pool.connection() as conn:
        await conn.execute("truncate documents, chunks restart identity cascade")
    yield pool
    await pool.close()


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()
