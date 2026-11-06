"""Client for reranking endpoints in the common /rerank format.

The format is served by llama.cpp's server, Jina, and Cohere-compatible gateways.

A reranker is a cross-encoder: it reads the query and one document together and scores
how well the document answers the query. That is more accurate than comparing two
independently computed embeddings, and too slow to run on the whole corpus, so it is
applied only to the top candidates of the first-stage retrieval.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import httpx

from .providers import ProviderError, post_json


class RerankError(ProviderError):
    """The reranking provider failed or returned an unusable response."""

    kind = "reranking"


class Reranker(Protocol):
    model: str

    async def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Returns one relevance score per document, in input order. Higher is more relevant."""
        ...


class HttpReranker:
    """Client for a /rerank endpoint (llama.cpp server, Jina, Cohere-compatible gateways).

    The request carries the query and the documents, and the response lists a relevance
    score per document index. OpenAI has no reranking API, so unlike the embedder this
    client does not follow an OpenAI format.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
        client: httpx.AsyncClient | None = None,
        max_attempts: int = 3,
    ) -> None:
        self.model = model
        self._url = base_url.rstrip("/") + "/rerank"
        self._api_key = api_key
        self._max_attempts = max_attempts
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)
        self.prompt_tokens = 0

    async def aclose(self) -> None:
        await self._client.aclose()

    async def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        payload = {"model": self.model, "query": query, "documents": list(documents), "top_n": len(documents)}
        response = await post_json(
            self._client, self._url, payload, self._api_key, RerankError, self._max_attempts
        )
        try:
            body = response.json()
            results = body["results"]
            scores: dict[int, float] = {int(r["index"]): float(r["relevance_score"]) for r in results}
        except (ValueError, KeyError, TypeError) as error:
            raise RerankError("reranking response did not have the expected shape") from error
        if sorted(scores) != list(range(len(documents))):
            raise RerankError(f"expected scores for {len(documents)} documents, received {len(scores)}")
        usage = body.get("usage") or {}
        self.prompt_tokens += int(usage.get("prompt_tokens", 0))
        return [scores[i] for i in range(len(documents))]
