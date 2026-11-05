"""Client for OpenAI-compatible embedding endpoints (llama.cpp server, OpenAI, Gemini, LiteLLM)."""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from collections.abc import Sequence
from typing import Protocol

import httpx

from .config import Settings

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


class EmbeddingError(Exception):
    """The embedding provider failed or returned an unusable response."""


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
        query_cache_size: int = 2048,
    ) -> None:
        self.model = settings.embedding_model
        self.dimensions = settings.embedding_dimensions
        self._settings = settings
        self._max_attempts = max_attempts
        self._client = client or httpx.AsyncClient(timeout=settings.embedding_timeout_seconds)
        self._query_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._query_cache_size = query_cache_size
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
        headers = (
            {"Authorization": f"Bearer {self._settings.embedding_api_key}"}
            if self._settings.embedding_api_key
            else {}
        )
        payload = {"model": self.model, "input": inputs}

        for attempt in range(1, self._max_attempts + 1):
            try:
                response = await self._client.post(url, json=payload, headers=headers)
            except httpx.TransportError as error:
                if attempt == self._max_attempts:
                    raise EmbeddingError(f"embedding request failed: {error}") from error
            else:
                if response.status_code == 200:
                    return self._parse(response, expected=len(inputs))
                if response.status_code not in RETRYABLE_STATUS or attempt == self._max_attempts:
                    raise EmbeddingError(f"embedding provider returned HTTP {response.status_code}")
            delay = 0.5 * 2 ** (attempt - 1)
            logger.warning("embedding request failed, retrying", extra={"attempt": attempt, "delay_s": delay})
            await asyncio.sleep(delay)
        raise AssertionError("unreachable")

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
