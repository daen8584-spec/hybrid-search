"""Command-line entry point: schema migration, serving, dataset ingestion, evaluation and benchmarks."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
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
from .embeddings import OpenAICompatibleEmbedder
from .evaluation.beir import CorpusDocument, iter_corpus, load_dataset, safe_extract
from .evaluation.metrics import mrr_at_k, ndcg_at_k, percentile, recall_at_k
from .ingest import Ingestor
from .log import configure_logging
from .search import SEARCH_MODES, SearchMode, SearchService

logger = logging.getLogger("hybrid_search.cli")

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    parser = _parser()
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    configure_logging(settings.log_level)
    try:
        result = args.handler(args, settings)
        if asyncio.iscoroutine(result):
            result = asyncio.run(result)
        return int(result or 0)
    except KeyboardInterrupt:
        return 130


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
    command.add_argument("--output", type=Path, default=Path("results"))
    command.set_defaults(handler=cmd_eval)

    command = commands.add_parser("ann-sweep", help="HNSW recall and latency against exact search")
    command.add_argument("dataset", type=Path)
    command.add_argument("--split", default="test")
    command.add_argument("--k", type=int, default=10)
    command.add_argument("--ef", default="10,20,40,80,160")
    command.add_argument("--output", type=Path, default=Path("results"))
    command.set_defaults(handler=cmd_ann_sweep)

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
        print(
            json.dumps({**counts, "seconds": round(elapsed, 1), "embedding_tokens": embedder.prompt_tokens})
        )
        return 1 if counts["failed"] else 0
    finally:
        await embedder.aclose()
        await pool.close()


async def cmd_eval(args: argparse.Namespace, settings: Settings) -> int:
    modes: list[SearchMode] = [m for m in SEARCH_MODES if m in args.modes.split(",")]
    dataset = load_dataset(args.dataset, args.split)
    pool = await open_pool(settings.database_url, max_size=4)
    embedder = OpenAICompatibleEmbedder(settings)
    try:
        await _check_dimensions(pool, embedder.dimensions)
        service = SearchService(pool, embedder)

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
        for mode in modes:
            totals = {"ndcg@10": 0.0, "recall@10": 0.0, "recall@100": 0.0, "mrr@10": 0.0}
            latencies = []
            for query in dataset.queries:
                started = time.perf_counter()
                ranked = await service.rank(query.text, mode, 100)
                latencies.append((time.perf_counter() - started) * 1000)
                ids = [doc.document_id for doc in ranked]
                relevant = dataset.qrels[query.id]
                totals["ndcg@10"] += ndcg_at_k(ids, relevant, 10)
                totals["recall@10"] += recall_at_k(ids, relevant, 10)
                totals["recall@100"] += recall_at_k(ids, relevant, 100)
                totals["mrr@10"] += mrr_at_k(ids, relevant, 10)
            count = len(dataset.queries)
            report["modes"][mode] = {
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
        _write(args.output, "eval", report)
        return 0
    finally:
        await embedder.aclose()
        await pool.close()


async def cmd_ann_sweep(args: argparse.Namespace, settings: Settings) -> int:
    ef_values = [int(v) for v in args.ef.split(",")]
    dataset = load_dataset(args.dataset, args.split)
    pool = await open_pool(settings.database_url, max_size=2)
    embedder = OpenAICompatibleEmbedder(settings)
    try:
        await _check_dimensions(pool, embedder.dimensions)
        service = SearchService(pool, embedder)
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
