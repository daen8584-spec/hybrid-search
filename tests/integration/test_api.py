import dataclasses
from collections.abc import Iterator, Sequence

import psycopg
import pytest
from fastapi.testclient import TestClient

from hybrid_search.api import create_app
from hybrid_search.config import Settings
from hybrid_search.embeddings import EmbeddingError

from .conftest import HashingEmbedder, TermOverlapReranker

pytestmark = pytest.mark.integration

DOCUMENT = {"title": "Aspirin", "text": "Aspirin reduces fever and inflammation.", "metadata": {"year": 2020}}


class FailingEmbedder(HashingEmbedder):
    async def embed_query(self, text: str) -> list[float]:
        raise EmbeddingError("provider is down")

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise EmbeddingError("provider is down")


def open_client(
    database_url: str, embedder: HashingEmbedder, reranker: TermOverlapReranker | None = None
) -> TestClient:
    with psycopg.connect(database_url, autocommit=True) as conn:
        conn.execute("truncate documents, chunks restart identity cascade")
    # A reranker configured in .env must not leak into tests that expect none.
    settings = dataclasses.replace(Settings.from_env(), database_url=database_url, rerank_base_url=None)
    return TestClient(create_app(settings, embedder=embedder, reranker=reranker))


@pytest.fixture
def client(database_url: str) -> Iterator[TestClient]:
    with open_client(database_url, HashingEmbedder()) as client:
        yield client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_put_creates_then_reports_unchanged(client: TestClient) -> None:
    created = client.put("/documents/doc-1", json=DOCUMENT)
    unchanged = client.put("/documents/doc-1", json=DOCUMENT)

    assert created.status_code == 201
    assert created.json() == {"id": "doc-1", "status": "created", "chunks": 1, "embedded": 1, "reused": 0}
    assert unchanged.status_code == 200
    assert unchanged.json()["status"] == "unchanged"


def test_put_validates_the_body_and_the_id(client: TestClient) -> None:
    assert client.put("/documents/doc-1", json={"title": "No text"}).status_code == 422
    assert client.put("/documents/doc-1", json={"text": "   "}).status_code == 422
    assert client.put("/documents/bad id", json=DOCUMENT).status_code == 422


def test_search_returns_ranked_hits_in_every_mode(client: TestClient) -> None:
    client.put("/documents/aspirin", json=DOCUMENT)
    client.put("/documents/sleep", json={"title": "Sleep", "text": "Sleep deprivation impairs memory."})

    for mode in ("keyword", "vector", "hybrid"):
        response = client.get("/search", params={"q": "aspirin fever", "mode": mode, "limit": 5})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["mode"] == mode
        assert body["results"][0]["id"] == "aspirin"
        assert body["results"][0]["title"] == "Aspirin"
        assert body["took_ms"] >= 0


def test_search_validates_parameters(client: TestClient) -> None:
    assert client.get("/search", params={"q": ""}).status_code == 422
    assert client.get("/search", params={"q": "x", "limit": 51}).status_code == 422
    assert client.get("/search", params={"q": "x", "mode": "fuzzy"}).status_code == 422


def test_delete(client: TestClient) -> None:
    client.put("/documents/doc-1", json=DOCUMENT)
    assert client.delete("/documents/doc-1").status_code == 204
    assert client.delete("/documents/doc-1").status_code == 404
    assert client.get("/search", params={"q": "aspirin", "mode": "keyword"}).json()["results"] == []


def test_request_ids_are_returned_and_propagated(client: TestClient) -> None:
    assert len(client.get("/health").headers["x-request-id"]) == 32
    assert client.get("/health", headers={"x-request-id": "abc-123"}).headers["x-request-id"] == "abc-123"


def test_embedding_failures_return_bad_gateway(database_url: str) -> None:
    with open_client(database_url, FailingEmbedder()) as client:
        assert client.put("/documents/doc-1", json=DOCUMENT).status_code == 502
        response = client.get("/search", params={"q": "aspirin", "mode": "vector"})
        assert response.status_code == 502
        assert response.json() == {"detail": "embedding provider unavailable"}
        # keyword search does not depend on the embedding provider
        assert client.get("/search", params={"q": "aspirin", "mode": "keyword"}).status_code == 200


def test_rerank_is_rejected_when_no_reranker_is_configured(client: TestClient) -> None:
    response = client.get("/search", params={"q": "aspirin", "rerank": "true"})
    assert response.status_code == 400
    assert response.json() == {"detail": "reranking is not configured"}


def test_search_with_reranking_reports_the_rerank_rank(database_url: str) -> None:
    with open_client(database_url, HashingEmbedder(), TermOverlapReranker()) as client:
        client.put("/documents/aspirin", json=DOCUMENT)
        client.put("/documents/sleep", json={"title": "Sleep", "text": "Sleep deprivation impairs memory."})

        body = client.get("/search", params={"q": "aspirin fever", "rerank": "true"}).json()

        assert body["reranked"] is True
        assert body["results"][0]["id"] == "aspirin"
        assert body["results"][0]["rerank_rank"] == 1
        plain = client.get("/search", params={"q": "aspirin fever"}).json()
        assert plain["reranked"] is False
        assert plain["results"][0]["rerank_rank"] is None
