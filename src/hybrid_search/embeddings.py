"""Client for OpenAI-compatible embedding endpoints (llama.cpp server, OpenAI, Gemini, LiteLLM)."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from typing import Protocol

import httpx

from .config import Settings
from .providers import ProviderError, post_json


class EmbeddingError(ProviderError):
    """The embedding provider failed or returned an unusable response."""

    kind = "embedding"


class Embedder(Protocol):
    model: str
    dimensions: int

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class OpenAICompatibleEmbedder:
    """Batches document embeddings, caches query embeddings and retries transient failures."""

    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient | None = None,
        max_attempts: int = 3,
        query_cache_size: int | None = None,
    ) -> None:
        self.model = settings.embedding_model
        self.dimensions = settings.embedding_dimensions
        self._settings = settings
        self._max_attempts = max_attempts
        self._client = client or httpx.AsyncClient(timeout=settings.embedding_timeout_seconds)
        self._query_cache: OrderedDict[str, list[float]] = OrderedDict()
        # Least recently used queries are evicted first. Size it above the number of
        # distinct queries that repeat, or a repeated pass in order evicts every entry
        # before it is reused.
        self._query_cache_size = (
            settings.embedding_query_cache_size if query_cache_size is None else query_cache_size
        )
        self.prompt_tokens = 0
        self.cache_hits = 0

    async def aclose(self) -> None:
        await self._client.aclose()

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        prefix = self._settings.embedding_document_prefix
        size = self._settings.embedding_batch_size
        vectors: list[list[float]] = []
        for start in range(0, len(texts), size):
            vectors.extend(await self._embed([prefix + t for t in texts[start : start + size]]))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        cached = self._query_cache.get(text)
        if cached is not None:
            self._query_cache.move_to_end(text)
            self.cache_hits += 1
            return cached
        [vector] = await self._embed([self._settings.embedding_query_prefix + text])
        self._query_cache[text] = vector
        if len(self._query_cache) > self._query_cache_size:
            self._query_cache.popitem(last=False)
        return vector

    async def _embed(self, inputs: list[str]) -> list[list[float]]:
        url = self._settings.embedding_base_url.rstrip("/") + "/embeddings"
        payload = {"model": self.model, "input": inputs}
        response = await post_json(
            self._client, url, payload, self._settings.embedding_api_key, EmbeddingError, self._max_attempts
        )
        return self._parse(response, expected=len(inputs))

    def _parse(self, response: httpx.Response, expected: int) -> list[list[float]]:
        try:
            body = response.json()
            items = sorted(body["data"], key=lambda item: item["index"])
            vectors = [[float(x) for x in item["embedding"]] for item in items]
        except (ValueError, KeyError, TypeError) as error:
            raise EmbeddingError("embedding response did not have the expected shape") from error

        if len(vectors) != expected:
            raise EmbeddingError(f"expected {expected} embeddings, received {len(vectors)}")
        for vector in vectors:
            if len(vector) != self.dimensions:
                raise EmbeddingError(f"expected {self.dimensions} dimensions, received {len(vector)}")
        usage = body.get("usage") or {}
        self.prompt_tokens += int(usage.get("prompt_tokens", 0))
        return vectors
