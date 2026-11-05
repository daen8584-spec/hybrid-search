import json
from collections.abc import Callable

import httpx
import pytest

from hybrid_search.config import Settings
from hybrid_search.embeddings import EmbeddingError, OpenAICompatibleEmbedder

pytestmark = pytest.mark.anyio

Handler = Callable[[httpx.Request], httpx.Response]


def settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "database_url": "postgresql:///unused",
        "embedding_base_url": "http://embeddings.test/v1",
        "embedding_model": "test-model",
        "embedding_api_key": "secret",
        "embedding_dimensions": 3,
        "embedding_query_prefix": "q: ",
        "embedding_document_prefix": "d: ",
        "embedding_batch_size": 2,
        "embedding_timeout_seconds": 5.0,
        "chunk_max_tokens": 384,
        "chunk_overlap_tokens": 48,
        "hnsw_ef_search": 40,
        "fusion_keyword_weight": 1.0,
        "rerank_base_url": None,
        "rerank_model": "test-reranker",
        "rerank_api_key": None,
        "rerank_depth": 20,
        "rerank_weight": 1.0,
        "log_level": "INFO",
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def embedder(handler: Handler, **overrides: object) -> OpenAICompatibleEmbedder:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAICompatibleEmbedder(settings(**overrides), client=client)


def vectors_for(request: httpx.Request) -> httpx.Response:
    inputs = json.loads(request.content)["input"]
    # Returned out of order on purpose: the client must sort by index.
    data = [{"index": i, "embedding": [float(len(text)), 0.0, 1.0]} for i, text in enumerate(inputs)][::-1]
    return httpx.Response(200, json={"data": data, "usage": {"prompt_tokens": 5 * len(inputs)}})


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    async def instant(_: float) -> None:
        return None

    monkeypatch.setattr("hybrid_search.providers.asyncio.sleep", instant)


async def test_documents_are_batched_prefixed_and_kept_in_order() -> None:
    requests: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content)["input"])
        assert request.headers["authorization"] == "Bearer secret"
        assert request.url == "http://embeddings.test/v1/embeddings"
        return vectors_for(request)

    client = embedder(handler)
    result = await client.embed_documents(["a", "bb", "ccc"])

    assert requests == [["d: a", "d: bb"], ["d: ccc"]]
    assert [v[0] for v in result] == [4.0, 5.0, 6.0]
    assert client.prompt_tokens == 15


async def test_query_embeddings_are_cached() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert json.loads(request.content)["input"] == ["q: hello"]
        return vectors_for(request)

    client = embedder(handler)
    first = await client.embed_query("hello")
    second = await client.embed_query("hello")

    assert first == second
    assert calls == 1
    assert client.cache_hits == 1


async def test_transient_failures_are_retried() -> None:
    responses = iter([httpx.Response(503), httpx.Response(429)])

    def handler(request: httpx.Request) -> httpx.Response:
        response = next(responses, None)
        return response if response is not None else vectors_for(request)

    assert len(await embedder(handler).embed_documents(["x"])) == 1


async def test_client_errors_are_not_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"error": "bad request"})

    with pytest.raises(EmbeddingError, match="HTTP 400"):
        await embedder(handler).embed_query("x")
    assert calls == 1


async def test_connection_errors_fail_after_the_last_attempt() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(EmbeddingError, match="request failed"):
        await embedder(handler).embed_query("x")
    assert calls == 3


async def test_a_dimension_mismatch_is_rejected() -> None:
    with pytest.raises(EmbeddingError, match="expected 4 dimensions"):
        await embedder(vectors_for, embedding_dimensions=4).embed_query("x")


async def test_a_malformed_response_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    with pytest.raises(EmbeddingError, match="expected shape"):
        await embedder(handler).embed_query("x")


async def test_no_authorization_header_without_a_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        return vectors_for(request)

    await embedder(handler, embedding_api_key=None).embed_query("x")
