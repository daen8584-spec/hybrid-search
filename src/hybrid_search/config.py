"""Runtime settings, read from environment variables (optionally loaded from a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: Path = Path(".env")) -> None:
    """Loads KEY=VALUE lines from a .env file. Variables already set in the environment win."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    try:
        return int(raw) if raw else default
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from error


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    try:
        return float(raw) if raw else default
    except ValueError as error:
        raise ValueError(f"{name} must be a number, got {raw!r}") from error


@dataclass(frozen=True, slots=True)
class Settings:
    database_url: str
    embedding_base_url: str
    embedding_model: str
    embedding_api_key: str | None
    embedding_dimensions: int
    embedding_query_prefix: str
    embedding_document_prefix: str
    embedding_batch_size: int
    embedding_timeout_seconds: float
    chunk_max_tokens: int
    chunk_overlap_tokens: int
    hnsw_ef_search: int
    fusion_keyword_weight: float
    rerank_base_url: str | None
    rerank_model: str
    rerank_api_key: str | None
    rerank_depth: int
    rerank_weight: float
    log_level: str

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ
        return cls(
            database_url=env.get("DATABASE_URL", "postgresql:///hybrid_search"),
            embedding_base_url=env.get("EMBEDDING_BASE_URL", "http://127.0.0.1:8081/v1"),
            embedding_model=env.get("EMBEDDING_MODEL", "nomic-embed-text-v1.5"),
            embedding_api_key=env.get("EMBEDDING_API_KEY") or None,
            embedding_dimensions=_int("EMBEDDING_DIMENSIONS", 768),
            embedding_query_prefix=env.get("EMBEDDING_QUERY_PREFIX", "search_query: "),
            embedding_document_prefix=env.get("EMBEDDING_DOCUMENT_PREFIX", "search_document: "),
            embedding_batch_size=_int("EMBEDDING_BATCH_SIZE", 16),
            embedding_timeout_seconds=_float("EMBEDDING_TIMEOUT_SECONDS", 60.0),
            chunk_max_tokens=_int("CHUNK_MAX_TOKENS", 384),
            chunk_overlap_tokens=_int("CHUNK_OVERLAP_TOKENS", 48),
            hnsw_ef_search=_int("HNSW_EF_SEARCH", 40),
            fusion_keyword_weight=_float("FUSION_KEYWORD_WEIGHT", 1.0),
            rerank_base_url=env.get("RERANK_BASE_URL") or None,
            rerank_model=env.get("RERANK_MODEL", "jina-reranker-v1-turbo-en"),
            rerank_api_key=env.get("RERANK_API_KEY") or None,
            rerank_depth=_int("RERANK_DEPTH", 20),
            rerank_weight=_float("RERANK_WEIGHT", 1.0),
            log_level=env.get("LOG_LEVEL", "INFO").upper(),
        )
