"""Measures the shared query embedding cache with several API-like processes.

Each process has its own embedder and in-memory cache, as an API worker does, and embeds a
stream of queries drawn from a Zipf distribution, so popular queries repeat. A stub
embedding provider answers after a fixed delay. The script reports provider calls and the
latency of embed_query with and without the shared PostgreSQL cache.

    python scripts/bench_query_cache.py --processes 4 --queries 2000 --distinct 5000

Requires DATABASE_URL with the migrations applied. The cache table is emptied first.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing
import os
import random
import statistics
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from hybrid_search.config import Settings
from hybrid_search.db import open_pool
from hybrid_search.embeddings import OpenAICompatibleEmbedder
from hybrid_search.query_cache import PostgresQueryCache

DIMENSIONS = 768


def stub_provider(latency: float) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            time.sleep(latency)
            data = [
                {"index": i, "embedding": [float(len(text) % 7)] * DIMENSIONS}
                for i, text in enumerate(body["input"])
            ]
            payload = json.dumps({"data": data}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def zipf_stream(seed: int, queries: int, distinct: int, exponent: float) -> list[str]:
    weights = [1.0 / (rank**exponent) for rank in range(1, distinct + 1)]
    picks = random.Random(seed).choices(range(distinct), weights=weights, k=queries)
    return [f"query number {i}" for i in picks]


async def run_process(
    settings: Settings, shared: bool, stream: list[str], concurrency: int
) -> dict[str, Any]:
    pool = await open_pool(settings.database_url, max_size=concurrency)
    embedder = OpenAICompatibleEmbedder(settings)
    if shared:
        embedder.shared_cache = PostgresQueryCache(
            pool, settings.embedding_model, DIMENSIONS, ttl_seconds=3600.0, max_rows=1_000_000
        )
    latencies: list[float] = []
    queue: asyncio.Queue[str] = asyncio.Queue()
    for query in stream:
        queue.put_nowait(query)

    async def worker() -> None:
        while not queue.empty():
            query = queue.get_nowait()
            start = time.perf_counter()
            await embedder.embed_query(query)
            latencies.append((time.perf_counter() - start) * 1000)

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    await embedder.aclose()
    await pool.close()
    return {
        "latencies": latencies,
        "provider_calls": embedder.provider_query_calls,
        "local_hits": embedder.cache_hits,
        "shared_hits": embedder.shared_cache_hits,
    }


def process_main(args: tuple[Settings, bool, list[str], int]) -> dict[str, Any]:
    return asyncio.run(run_process(*args))


async def empty_cache(url: str) -> None:
    pool = await open_pool(url, max_size=1)
    async with pool.connection() as conn:
        await conn.execute("truncate query_embedding_cache")
    await pool.close()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--queries", type=int, default=2000, help="queries per process")
    parser.add_argument("--distinct", type=int, default=5000, help="distinct queries in the distribution")
    parser.add_argument("--exponent", type=float, default=1.0, help="Zipf exponent")
    parser.add_argument("--concurrency", type=int, default=8, help="concurrent queries per process")
    parser.add_argument("--latency-ms", type=float, default=150.0, help="stub provider latency")
    args = parser.parse_args()

    server = stub_provider(args.latency_ms / 1000)
    base = Settings.from_env()
    settings = replace(
        base,
        embedding_base_url=f"http://127.0.0.1:{server.server_port}/v1",
        embedding_model="stub-model",
        embedding_dimensions=DIMENSIONS,
        embedding_api_key=None,
    )
    streams = [
        zipf_stream(seed, args.queries, args.distinct, args.exponent) for seed in range(args.processes)
    ]
    print(
        f"{args.processes} processes x {args.queries} queries, Zipf exponent {args.exponent} over "
        f"{args.distinct} distinct queries, {args.concurrency} concurrent per process, "
        f"provider latency {args.latency_ms:.0f} ms, "
        f"local cache {settings.embedding_query_cache_size} entries"
    )
    print()
    print("| Shared cache | Provider calls | Hit rate | Mean ms | p95 ms | p99 ms | Wall s |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    context = multiprocessing.get_context("spawn")
    for shared in (False, True):
        asyncio.run(empty_cache(settings.database_url))
        start = time.perf_counter()
        with context.Pool(args.processes) as workers:
            results = workers.map(
                process_main, [(settings, shared, stream, args.concurrency) for stream in streams]
            )
        wall = time.perf_counter() - start
        latencies = [value for result in results for value in result["latencies"]]
        calls = sum(result["provider_calls"] for result in results)
        total = args.processes * args.queries
        print(
            f"| {'yes' if shared else 'no'} | {calls:,} | {1 - calls / total:.1%} | "
            f"{statistics.fmean(latencies):.1f} | {percentile(latencies, 0.95):.1f} | "
            f"{percentile(latencies, 0.99):.1f} | {wall:.1f} |"
        )
    server.shutdown()


if __name__ == "__main__":
    os.environ.setdefault("LOG_LEVEL", "WARNING")
    main()
