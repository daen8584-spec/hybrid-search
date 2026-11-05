"""HTTP API: document upsert and delete, search, and a health check."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Path, Query, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import Settings
from .db import embedding_dimensions, open_pool
from .embeddings import Embedder, OpenAICompatibleEmbedder
from .ingest import Ingestor, IngestStatus
from .log import configure_logging, request_id
from .providers import ProviderError
from .reranker import OpenAICompatibleReranker, Reranker
from .search import SearchMode, SearchService

logger = logging.getLogger("hybrid_search.api")

DocumentId = Annotated[str, Path(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9._:-]+$")]


class DocumentIn(BaseModel):
    title: str = Field(default="", max_length=1_000)
    text: str = Field(min_length=1, max_length=1_000_000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentOut(BaseModel):
    id: str
    status: IngestStatus
    chunks: int
    embedded: int
    reused: int


class Hit(BaseModel):
    id: str
    title: str
    snippet: str
    score: float
    keyword_rank: int | None
    vector_rank: int | None
    rerank_rank: int | None = None


class SearchOut(BaseModel):
    query: str
    mode: SearchMode
    reranked: bool
    took_ms: float
    results: list[Hit]


def create_app(
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.log_level)
        pool = await open_pool(settings.database_url)
        owned = embedder is None
        active: Embedder = embedder or OpenAICompatibleEmbedder(settings)
        owned_reranker = reranker is None and settings.rerank_base_url is not None
        active_reranker: Reranker | None = reranker
        if owned_reranker and settings.rerank_base_url:
            active_reranker = OpenAICompatibleReranker(
                settings.rerank_base_url,
                settings.rerank_model,
                settings.rerank_api_key,
                settings.embedding_timeout_seconds,
            )
        try:
            async with pool.connection() as conn:
                declared = await embedding_dimensions(conn)
            if declared != active.dimensions:
                raise RuntimeError(
                    f"chunks.embedding has {declared} dimensions but the embedding model produces "
                    f"{active.dimensions}; see docs/adr/0004 for changing models"
                )
            app.state.pool = pool
            app.state.ingestor = Ingestor(
                pool, active, settings.chunk_max_tokens, settings.chunk_overlap_tokens
            )
            app.state.search = SearchService.from_settings(pool, active, settings, active_reranker)
            yield
        finally:
            if owned and isinstance(active, OpenAICompatibleEmbedder):
                await active.aclose()
            if owned_reranker and isinstance(active_reranker, OpenAICompatibleReranker):
                await active_reranker.aclose()
            await pool.close()

    app = FastAPI(title="hybrid-search", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        current = request.headers.get("x-request-id") or uuid.uuid4().hex
        token = request_id.set(current)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("unhandled error", extra={"method": request.method, "path": request.url.path})
            response = JSONResponse({"detail": "internal server error"}, status_code=500)
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        response.headers["x-request-id"] = current
        logger.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        request_id.reset(token)
        return response

    @app.exception_handler(ProviderError)
    async def provider_failed(request: Request, error: ProviderError) -> JSONResponse:
        logger.error(f"{error.kind} provider error", extra={"detail": str(error)})
        return JSONResponse(
            {"detail": f"{error.kind} provider unavailable"}, status_code=status.HTTP_502_BAD_GATEWAY
        )

    @app.get("/health")
    async def health(request: Request) -> dict[str, str]:
        async with request.app.state.pool.connection() as conn:
            await conn.execute("select 1")
        return {"status": "ok"}

    @app.put("/documents/{document_id}", response_model=DocumentOut)
    async def put_document(
        document_id: DocumentId, body: DocumentIn, request: Request, response: Response
    ) -> DocumentOut:
        if not body.text.strip():
            raise HTTPException(422, "text must not be blank")
        result = await request.app.state.ingestor.upsert(document_id, body.title, body.text, body.metadata)
        if result.status == "created":
            response.status_code = status.HTTP_201_CREATED
        return DocumentOut(
            id=result.document_id,
            status=result.status,
            chunks=result.chunks,
            embedded=result.embedded,
            reused=result.reused,
        )

    @app.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_document(document_id: DocumentId, request: Request) -> Response:
        if not await request.app.state.ingestor.delete(document_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get("/search", response_model=SearchOut)
    async def search(
        request: Request,
        q: Annotated[str, Query(min_length=1, max_length=1_000)],
        mode: SearchMode = "hybrid",
        limit: Annotated[int, Query(ge=1, le=50)] = 10,
        rerank: bool = False,
    ) -> SearchOut:
        service: SearchService = request.app.state.search
        if rerank and service.reranker is None:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "reranking is not configured")
        started = time.perf_counter()
        hits = await service.search(q, mode, limit, rerank)
        return SearchOut(
            query=q,
            mode=mode,
            reranked=rerank,
            took_ms=round((time.perf_counter() - started) * 1000, 2),
            results=[
                Hit(
                    id=h.document_id,
                    title=h.title,
                    snippet=h.snippet,
                    score=h.score,
                    keyword_rank=h.keyword_rank,
                    vector_rank=h.vector_rank,
                    rerank_rank=h.rerank_rank,
                )
                for h in hits
            ],
        )

    return app
