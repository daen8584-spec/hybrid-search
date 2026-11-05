"""Command-line entry point: schema migration, serving, dataset ingestion, evaluation and benchmarks."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import random
import statistics
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Settings, load_dotenv
from .db import embedding_dimensions, migrate, open_pool
from .embeddings import EmbeddingError, OpenAICompatibleEmbedder
from .evaluation.beir import CorpusDocument, iter_corpus, load_dataset, safe_extract
from .evaluation.metrics import mrr_at_k, ndcg_at_k, percentile, recall_at_k
from .ingest import Ingestor
from .log import configure_logging
from .providers import ProviderError
from .reranker import OpenAICompatibleReranker
from .search import SEARCH_MODES, SearchMode, SearchService, apply_rerank, fuse

logger = logging.getLogger("hybrid_search.cli")

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    parser = _parser()
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    configure_logging(settings.log_level)
    from psycopg import OperationalError

    try:
        result = args.handler(args, settings)
        if asyncio.iscoroutine(result):
            result = asyncio.run(result)
        return int(result or 0)
    except KeyboardInterrupt:
        return 130
    except ProviderError as error:
        url = settings.embedding_base_url if isinstance(error, EmbeddingError) else settings.rerank_base_url
        print(f"error: {error} ({url})", file=sys.stderr)
        return 1
    except OperationalError as error:
        print(f"error: database unavailable: {error}".rstrip(), file=sys.stderr)
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hybrid-search")
    commands = parser.add_subparsers(required=True, metavar="command")

    command = commands.add_parser("migrate", help="apply pending schema migrations")
    command.set_defaults(handler=cmd_migrate)

    command = commands.add_parser("serve", help="run the HTTP API")
    command.add_argument("--host", default="127.0.0.1")
    command.add_argument("--port", type=int, default=8000)
    command.set_defaults(handler=cmd_serve)

    command = commands.add_parser("download-beir", help="download and extract a BEIR dataset")
    command.add_argument("--name", default="scifact")
    command.add_argument("--dir", type=Path, default=Path("data"))
    command.set_defaults(handler=cmd_download_beir)

    command = commands.add_parser("ingest-beir", help="ingest a BEIR corpus (resumable)")
    command.add_argument("dataset", type=Path, help="dataset directory, for example data/scifact")
    command.add_argument("--concurrency", type=int, default=2)
    command.add_argument("--limit", type=int, default=None, help="ingest only the first N documents")
    command.set_defaults(handler=cmd_ingest_beir)

    command = commands.add_parser("eval", help="measure retrieval quality and latency per search mode")
    command.add_argument("dataset", type=Path)
    command.add_argument("--split", default="test")
    command.add_argument("--modes", default=",".join(SEARCH_MODES))
    command.add_argument("--rerank", action="store_true", help="also evaluate hybrid search with reranking")
    command.add_argument("--output", type=Path, default=Path("results"))
    command.set_defaults(handler=cmd_eval)

    command = commands.add_parser("tune-rerank", help="hybrid quality with reranking for a range of weights")
    command.add_argument("dataset", type=Path)
    command.add_argument("--split", default="train")
    command.add_argument("--queries", type=int, default=300, help="random sample of queries from the split")
    command.add_argument("--weights", default="0,0.25,0.5,1,2,4,inf")
    command.add_argument("--output", type=Path, default=Path("results"))
    command.set_defaults(handler=cmd_tune_rerank)

    command = commands.add_parser("ann-sweep", help="HNSW recall and latency against exact search")
    command.add_argument("dataset", type=Path)
    command.add_argument("--split", default="test")
    command.add_argument("--k", type=int, default=10)
    command.add_argument("--ef", default="10,20,40,80,160")
    command.add_argument("--output", type=Path, default=Path("results"))
    command.set_defaults(handler=cmd_ann_sweep)

    command = commands.add_parser("tune-fusion", help="hybrid quality for a range of keyword weights")
    command.add_argument("dataset", type=Path)
    command.add_argument("--split", default="train")
    command.add_argument("--weights", default="0,0.1,0.2,0.3,0.4,0.5,0.7,1.0")
    command.add_argument("--output", type=Path, default=Path("results"))
    command.set_defaults(handler=cmd_tune_fusion)

    command = commands.add_parser("bench", help="load test a running API with dataset queries")
    command.add_argument("dataset", type=Path)
    command.add_argument("--url", default="http://127.0.0.1:8000")
    command.add_argument("--split", default="test")
    command.add_argument("--mode", choices=SEARCH_MODES, default="hybrid")
    command.add_argument("--concurrency", type=int, default=8)
    command.add_argument("--output", type=Path, default=Path("results"))
    command.set_defaults(handler=cmd_bench)
    return parser


async def cmd_migrate(args: argparse.Namespace, settings: Settings) -> int:
    from psycopg import AsyncConnection

    async with await AsyncConnection.connect(settings.database_url, autocommit=True) as conn:
        applied = await migrate(conn)
        if applied:
            # Migrations can backfill large tables; fresh statistics and visibility maps let
            # the planner use index-only scans on chunk_terms immediately.
            await conn.execute("vacuum (analyze) chunks, chunk_terms")
    print(f"applied: {', '.join(applied)}" if applied else "schema is up to date")
    return 0


def cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_config=None)
    return 0


async def cmd_download_beir(args: argparse.Namespace, settings: Settings) -> int:
    import httpx

    target = args.dir / args.name
    if (target / "corpus.jsonl").exists():
        print(f"{target} already exists")
        return 0
    args.dir.mkdir(parents=True, exist_ok=True)
    archive = args.dir / f"{args.name}.zip"
    url = BEIR_URL.format(name=args.name)
    async with (
        httpx.AsyncClient(timeout=120, follow_redirects=True) as client,
        client.stream("GET", url) as response,
    ):
        response.raise_for_status()
        with archive.open("wb") as handle:
            async for block in response.aiter_bytes():
                handle.write(block)
    safe_extract(archive, args.dir)
    archive.unlink()
    print(f"extracted to {target}")
    return 0


async def cmd_ingest_beir(args: argparse.Namespace, settings: Settings) -> int:
    pool = await open_pool(settings.database_url, max_size=args.concurrency + 1)
    embedder = OpenAICompatibleEmbedder(settings)
    try:
        await _check_dimensions(pool, embedder.dimensions)
        ingestor = Ingestor(pool, embedder, settings.chunk_max_tokens, settings.chunk_overlap_tokens)
        documents = list(iter_corpus(args.dataset))[: args.limit]
        counts = {
            "created": 0,
            "updated": 0,
            "unchanged": 0,
            "skipped": 0,
            "failed": 0,
            "chunks": 0,
            "embedded": 0,
        }
        semaphore = asyncio.Semaphore(args.concurrency)
        started = time.perf_counter()
        done = 0

        async def ingest(document: CorpusDocument) -> None:
            nonlocal done
            try:
                if not document.text.strip():
                    counts["skipped"] += 1
                    return
                async with semaphore:
                    result = await ingestor.upsert(
                        document.id, document.title, document.text, document.metadata
                    )
            except Exception:
                # Logged and counted so that one bad document does not stop a long run; rerunning
                # the command retries only the documents that were not stored.
                logger.exception("failed to ingest document", extra={"document_id": document.id})
                counts["failed"] += 1
            else:
                counts[result.status] += 1
                counts["chunks"] += result.chunks
                counts["embedded"] += result.embedded
            finally:
                done += 1
                if done % 100 == 0 or done == len(documents):
                    rate = done / (time.perf_counter() - started)
                    eta = (len(documents) - done) / rate if rate else 0
                    print(
                        f"{done}/{len(documents)} documents, {rate:.1f}/s, eta {eta / 60:.1f} min", flush=True
                    )

        await asyncio.gather(*(ingest(document) for document in documents))
        elapsed = time.perf_counter() - started
        await _vacuum(settings.database_url)
        print(
            json.dumps({**counts, "seconds": round(elapsed, 1), "embedding_tokens": embedder.prompt_tokens})
        )
        return 1 if counts["failed"] else 0
    finally:
        await embedder.aclose()
        await pool.close()


async def cmd_eval(args: argparse.Namespace, settings: Settings) -> int:
    runs: list[tuple[str, SearchMode, bool]] = [
        (mode, mode, False) for mode in SEARCH_MODES if mode in args.modes.split(",")
    ]
    if args.rerank:
        runs.append(("hybrid+rerank", "hybrid", True))
    dataset = load_dataset(args.dataset, args.split)
    pool = await open_pool(settings.database_url, max_size=4)
    embedder = OpenAICompatibleEmbedder(settings)
    reranker = _reranker(settings) if args.rerank else None
    try:
        await _check_dimensions(pool, embedder.dimensions)
        service = SearchService.from_settings(pool, embedder, settings, reranker)

        # Query embeddings are computed once up front, so retrieval latency is measured
        # separately from the embedding call and is comparable across modes.
        embedding_ms = []
        for query in dataset.queries:
            started = time.perf_counter()
            await embedder.embed_query(query.text)
            embedding_ms.append((time.perf_counter() - started) * 1000)

        report: dict[str, Any] = {
            "dataset": dataset.name,
            "split": args.split,
            "queries": len(dataset.queries),
            "embedding_model": settings.embedding_model,
            "query_embedding_ms": _latency(embedding_ms),
            "modes": {},
        }
        if reranker is not None:
            report["rerank"] = {
                "model": reranker.model,
                "depth": service.rerank_depth,
                "weight": service.rerank_weight,
            }
        for label, mode, rerank in runs:
            totals = {"ndcg@10": 0.0, "recall@10": 0.0, "recall@100": 0.0, "mrr@10": 0.0}
            latencies = []
            for query in dataset.queries:
                started = time.perf_counter()
                ranked = await service.rank(query.text, mode, 100, rerank)
                latencies.append((time.perf_counter() - started) * 1000)
                ids = [doc.document_id for doc in ranked]
                relevant = dataset.qrels[query.id]
                totals["ndcg@10"] += ndcg_at_k(ids, relevant, 10)
                totals["recall@10"] += recall_at_k(ids, relevant, 10)
                totals["recall@100"] += recall_at_k(ids, relevant, 100)
                totals["mrr@10"] += mrr_at_k(ids, relevant, 10)
            count = len(dataset.queries)
            report["modes"][label] = {
                **{name: round(total / count, 4) for name, total in totals.items()},
                "latency_ms": _latency(latencies),
            }

        print(f"\n{dataset.name} ({args.split}), {len(dataset.queries)} queries\n")
        print("| Mode | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | p50 ms | p95 ms |")
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for mode, row in report["modes"].items():
            print(
                f"| {mode} | {row['ndcg@10']:.3f} | {row['recall@10']:.3f} | {row['recall@100']:.3f} "
                f"| {row['mrr@10']:.3f} | {row['latency_ms']['p50']:.1f} | {row['latency_ms']['p95']:.1f} |"
            )
        print(f"\nQuery embedding: p50 {report['query_embedding_ms']['p50']:.1f} ms, not included above")
        if reranker is not None:
            print(
                f"Reranking: {reranker.model}, top {service.rerank_depth}, weight {service.rerank_weight}; "
                "its latency is included above"
            )
        _write(args.output, "eval", report)
        return 0
    finally:
        await embedder.aclose()
        if reranker is not None:
            await reranker.aclose()
        await pool.close()


async def cmd_tune_rerank(args: argparse.Namespace, settings: Settings) -> int:
    """Evaluates reranking weights on a split other than the one reported.

    Each query is retrieved and reranked once; every weight is then applied to the same
    scores, so the sweep costs one reranker call per query regardless of the number of
    weights.
    """
    weights = [float(v) for v in args.weights.split(",")]
    dataset = load_dataset(args.dataset, args.split)
    queries = list(dataset.queries)
    if args.queries < len(queries):
        queries = random.Random(7).sample(queries, args.queries)
    pool = await open_pool(settings.database_url, max_size=4)
    embedder = OpenAICompatibleEmbedder(settings)
    reranker = _reranker(settings)
    try:
        await _check_dimensions(pool, embedder.dimensions)
        service = SearchService.from_settings(pool, embedder, settings, reranker)
        depth = service.rerank_depth
        scored = []
        rerank_ms = []
        for index, query in enumerate(queries, start=1):
            ranked = await service.rank(query.text, "hybrid", 100)
            started = time.perf_counter()
            scores = await service.rerank_scores(query.text, ranked[:depth])
            rerank_ms.append((time.perf_counter() - started) * 1000)
            scored.append((query, ranked, scores))
            if index % 25 == 0 or index == len(queries):
                print(f"{index}/{len(queries)} queries reranked", flush=True)

        rows = []
        for weight in weights:
            ndcg, recall = 0.0, 0.0
            for query, ranked, scores in scored:
                ids = [doc.document_id for doc in apply_rerank(ranked, scores, weight, service.rrf_k)]
                ndcg += ndcg_at_k(ids, dataset.qrels[query.id], 10)
                recall += recall_at_k(ids, dataset.qrels[query.id], 10)
            rows.append(
                {
                    "rerank_weight": weight,
                    "ndcg@10": round(ndcg / len(scored), 4),
                    "recall@10": round(recall / len(scored), 4),
                }
            )

        print(f"\n{dataset.name} ({args.split}), {len(scored)} queries, {reranker.model}, top {depth}\n")
        print("| Rerank weight | nDCG@10 | Recall@10 |")
        print("| ---: | ---: | ---: |")
        for row in rows:
            label = "reranker order" if math.isinf(row["rerank_weight"]) else row["rerank_weight"]
            print(f"| {label} | {row['ndcg@10']:.3f} | {row['recall@10']:.3f} |")
        print(f"\nReranking latency for {depth} documents: p50 {percentile(rerank_ms, 50):.0f} ms")
        best = max(rows, key=lambda row: row["ndcg@10"])
        print(f"best weight by nDCG@10: {best['rerank_weight']}")
        _write(
            args.output,
            "tune-rerank",
            {
                "dataset": dataset.name,
                "split": args.split,
                "model": reranker.model,
                "depth": depth,
                "rerank_ms": _latency(rerank_ms),
                "rows": rows,
            },
        )
        return 0
    finally:
        await embedder.aclose()
        await reranker.aclose()
        await pool.close()


async def cmd_ann_sweep(args: argparse.Namespace, settings: Settings) -> int:
    ef_values = [int(v) for v in args.ef.split(",")]
    dataset = load_dataset(args.dataset, args.split)
    pool = await open_pool(settings.database_url, max_size=2)
    embedder = OpenAICompatibleEmbedder(settings)
    try:
        await _check_dimensions(pool, embedder.dimensions)
        service = SearchService.from_settings(pool, embedder, settings)
        vectors = [await embedder.embed_query(query.text) for query in dataset.queries]

        async def run(**options: Any) -> tuple[list[list[str]], list[float]]:
            results, latencies = [], []
            for vector in vectors:
                started = time.perf_counter()
                candidates = await service.vector(vector, args.k, **options)
                latencies.append((time.perf_counter() - started) * 1000)
                results.append([c.document_id for c in candidates])
            return results, latencies

        def ndcg(results: list[list[str]]) -> float:
            scores = [
                ndcg_at_k(ids, dataset.qrels[q.id], 10)
                for ids, q in zip(results, dataset.queries, strict=True)
            ]
            return statistics.fmean(scores)

        exact, exact_latency = await run(exact=True)
        rows = [
            {"ef_search": "exact", "recall": 1.0, "ndcg@10": round(ndcg(exact), 4), **_latency(exact_latency)}
        ]
        for ef in ef_values:
            approximate, latency = await run(ef_search=ef)
            recall = statistics.fmean(
                len(set(a).intersection(e)) / len(e) if e else 1.0
                for a, e in zip(approximate, exact, strict=True)
            )
            rows.append(
                {
                    "ef_search": ef,
                    "recall": round(recall, 4),
                    "ndcg@10": round(ndcg(approximate), 4),
                    **_latency(latency),
                }
            )

        print(f"\nHNSW recall@{args.k} against exact search, {len(vectors)} queries\n")
        print(f"| ef_search | Recall@{args.k} | nDCG@10 | p50 ms | p95 ms |")
        print("| ---: | ---: | ---: | ---: | ---: |")
        for row in rows:
            print(
                f"| {row['ef_search']} | {row['recall']:.3f} | {row['ndcg@10']:.3f} "
                f"| {row['p50']:.2f} | {row['p95']:.2f} |"
            )
        _write(args.output, "ann-sweep", {"dataset": dataset.name, "k": args.k, "rows": rows})
        return 0
    finally:
        await embedder.aclose()
        await pool.close()


async def cmd_tune_fusion(args: argparse.Namespace, settings: Settings) -> int:
    """Evaluates RRF keyword weights (vector weight 1.0) on a split other than the one reported.

    Both candidate lists are retrieved once per query and fused for every weight, so the
    sweep costs one retrieval per query regardless of the number of weights.
    """
    keyword_weights = [float(v) for v in args.weights.split(",")]
    dataset = load_dataset(args.dataset, args.split)
    pool = await open_pool(settings.database_url, max_size=4)
    embedder = OpenAICompatibleEmbedder(settings)
    try:
        await _check_dimensions(pool, embedder.dimensions)
        service = SearchService.from_settings(pool, embedder, settings)
        retrieved = []
        for query in dataset.queries:
            keyword = await service.keyword(query.text, service.candidates)
            vector = await service.vector(await embedder.embed_query(query.text), service.candidates)
            retrieved.append((query, keyword, vector))

        rows = []
        for weight in keyword_weights:
            ndcg, recall = 0.0, 0.0
            for query, keyword, vector in retrieved:
                fused = fuse(keyword, vector, 100, service.rrf_k, {"keyword": weight, "vector": 1.0})
                ids = [doc.document_id for doc in fused]
                ndcg += ndcg_at_k(ids, dataset.qrels[query.id], 10)
                recall += recall_at_k(ids, dataset.qrels[query.id], 100)
            count = len(retrieved)
            rows.append(
                {
                    "keyword_weight": weight,
                    "ndcg@10": round(ndcg / count, 4),
                    "recall@100": round(recall / count, 4),
                }
            )

        print(f"\n{dataset.name} ({args.split}), {len(dataset.queries)} queries, vector weight 1.0\n")
        print("| Keyword weight | nDCG@10 | Recall@100 |")
        print("| ---: | ---: | ---: |")
        for row in rows:
            print(f"| {row['keyword_weight']} | {row['ndcg@10']:.3f} | {row['recall@100']:.3f} |")
        best = max(rows, key=lambda row: row["ndcg@10"])
        print(f"\nbest keyword weight by nDCG@10: {best['keyword_weight']}")
        _write(args.output, "tune-fusion", {"dataset": dataset.name, "split": args.split, "rows": rows})
        return 0
    finally:
        await embedder.aclose()
        await pool.close()


async def cmd_bench(args: argparse.Namespace, settings: Settings) -> int:
    import httpx

    dataset = load_dataset(args.dataset, args.split)
    queries = [query.text for query in dataset.queries]
    random.Random(7).shuffle(queries)

    async def run_pass(client: httpx.AsyncClient) -> dict[str, Any]:
        latencies: list[float] = []
        errors = 0
        pending = iter(queries)

        async def worker() -> None:
            nonlocal errors
            for text in pending:
                started = time.perf_counter()
                try:
                    response = await client.get("/search", params={"q": text, "mode": args.mode, "limit": 10})
                    response.raise_for_status()
                except httpx.HTTPError:
                    errors += 1
                    continue
                latencies.append((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(args.concurrency)))
        elapsed = time.perf_counter() - started
        return {
            "requests": len(queries),
            "errors": errors,
            "throughput_rps": round(len(latencies) / elapsed, 1),
            **(_latency(latencies, with_p99=True) if latencies else {}),
        }

    limits = httpx.Limits(max_connections=args.concurrency)
    async with httpx.AsyncClient(base_url=args.url, timeout=60, limits=limits) as client:
        (await client.get("/health")).raise_for_status()
        # The first pass embeds every query; the second is served from the query embedding cache.
        cold = await run_pass(client)
        warm = await run_pass(client)

    report = {"mode": args.mode, "concurrency": args.concurrency, "cold": cold, "warm": warm}
    print(f"\n{args.mode} search, concurrency {args.concurrency}, {len(queries)} requests per pass\n")
    print("| Pass | Throughput (req/s) | p50 ms | p95 ms | p99 ms | Errors |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name, row in (("cold (query embedded)", cold), ("warm (embedding cached)", warm)):
        print(
            f"| {name} | {row['throughput_rps']} | {row.get('p50', 0):.1f} | {row.get('p95', 0):.1f} "
            f"| {row.get('p99', 0):.1f} | {row['errors']} |"
        )
    _write(args.output, "bench", report)
    return 0 if cold["errors"] == warm["errors"] == 0 else 1


def _reranker(settings: Settings) -> OpenAICompatibleReranker:
    if not settings.rerank_base_url:
        raise SystemExit("reranking needs RERANK_BASE_URL, for example http://127.0.0.1:8082/v1")
    return OpenAICompatibleReranker(
        settings.rerank_base_url,
        settings.rerank_model,
        settings.rerank_api_key,
        settings.embedding_timeout_seconds,
    )


async def _vacuum(url: str) -> None:
    from psycopg import AsyncConnection

    async with await AsyncConnection.connect(url, autocommit=True) as conn:
        await conn.execute("vacuum (analyze) chunks, chunk_terms")


async def _check_dimensions(pool: Any, expected: int) -> None:
    async with pool.connection() as conn:
        declared = await embedding_dimensions(conn)
    if declared != expected:
        raise SystemExit(f"chunks.embedding has {declared} dimensions, EMBEDDING_DIMENSIONS is {expected}")


def _latency(values: Sequence[float], with_p99: bool = False) -> dict[str, float]:
    summary = {
        "p50": round(percentile(values, 50), 2),
        "p95": round(percentile(values, 95), 2),
        "mean": round(statistics.fmean(values), 2),
    }
    if with_p99:
        summary["p99"] = round(percentile(values, 99), 2)
    return summary


def _write(directory: Path, kind: str, report: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"{kind}-{stamp}.json"
    path.write_text(json.dumps({"generated_at": stamp, **report}, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    sys.exit(main())
