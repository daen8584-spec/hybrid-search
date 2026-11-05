"""Shared HTTP handling for model providers (embeddings and reranking)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


class ProviderError(Exception):
    """A model provider failed or returned an unusable response."""

    kind = "model"


async def post_json(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    api_key: str | None,
    error: type[ProviderError],
    max_attempts: int = 3,
) -> httpx.Response:
    """POSTs JSON and returns a 200 response, retrying rate limits, server errors and
    connection failures with exponential backoff. Raises `error` otherwise."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    for attempt in range(1, max_attempts + 1):
        try:
            response = await client.post(url, json=payload, headers=headers)
        except httpx.TransportError as cause:
            if attempt == max_attempts:
                raise error(f"{error.kind} request failed: {cause}") from cause
        else:
            if response.status_code == 200:
                return response
            if response.status_code not in RETRYABLE_STATUS or attempt == max_attempts:
                raise error(f"{error.kind} provider returned HTTP {response.status_code}")
        delay = 0.5 * 2 ** (attempt - 1)
        logger.warning(f"{error.kind} request failed, retrying", extra={"attempt": attempt, "delay_s": delay})
        await asyncio.sleep(delay)
    raise AssertionError("unreachable")
