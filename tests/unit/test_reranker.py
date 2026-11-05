import json

import httpx
import pytest

from hybrid_search.reranker import OpenAICompatibleReranker, RerankError

pytestmark = pytest.mark.anyio


def reranker(handler, **options) -> OpenAICompatibleReranker:  # type: ignore[no-untyped-def]
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAICompatibleReranker("http://rerank.test/v1", "test-reranker", client=client, **options)


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    async def instant(_: float) -> None:
        return None

    monkeypatch.setattr("hybrid_search.providers.asyncio.sleep", instant)


async def test_scores_are_returned_in_input_order() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url == "http://rerank.test/v1/rerank"
        assert body == {"model": "test-reranker", "query": "q", "documents": ["a", "b", "c"], "top_n": 3}
        # Providers sort results by score, not by input position.
        results = [{"index": 2, "relevance_score": 3.0}, {"index": 0, "relevance_score": 1.5}]
        results.append({"index": 1, "relevance_score": -2.0})
        return httpx.Response(200, json={"results": results, "usage": {"prompt_tokens": 42}})

    client = reranker(handler)
    assert await client.score("q", ["a", "b", "c"]) == [1.5, -2.0, 3.0]
    assert client.prompt_tokens == 42


async def test_no_request_for_no_documents() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    assert await reranker(handler).score("q", []) == []


async def test_missing_scores_are_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 1.0}]})

    with pytest.raises(RerankError, match="expected scores for 2 documents"):
        await reranker(handler).score("q", ["a", "b"])


async def test_transient_errors_are_retried_and_others_reported() -> None:
    responses = iter([httpx.Response(503)])

    def flaky(request: httpx.Request) -> httpx.Response:
        response = next(responses, None)
        if response is not None:
            return response
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.5}]})

    assert await reranker(flaky).score("q", ["a"]) == [0.5]

    def broken(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401)

    with pytest.raises(RerankError, match="reranking provider returned HTTP 401"):
        await reranker(broken).score("q", ["a"])
