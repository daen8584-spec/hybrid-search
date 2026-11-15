# 0009. Share query embeddings between API processes through a PostgreSQL cache

- Status: Accepted
- Date: 2026-11-15

## Context

Embedding a query is the slowest step of a vector or hybrid search: 100 to 400 ms per query with the local model, against about 20 ms for retrieval and fusion. Each API process keeps an in-memory least-recently-used cache of query embeddings, so a repeated query is embedded once per process. With several processes behind a load balancer, a popular query is still embedded once in every process, and every process restart starts cold.

A shared cache needs a store that all processes reach. Redis is the usual choice, but this service keeps its documents, chunks, inverted index and vectors in PostgreSQL alone, so that one datastore is backed up, monitored and migrated ([0004](0004-embedding-model.md), [0006](0006-synchronous-ingestion.md)).

## Decision

- **A second-level cache in PostgreSQL**, in front of the provider and behind each process's in-memory cache: in-memory first, then PostgreSQL, then the provider. Enabled with `EMBEDDING_SHARED_CACHE=true`.
- **An unlogged table**, `query_embedding_cache`, keyed by model name and SHA-256 of the exact text sent to the provider, query prefix included. An unlogged table skips the write-ahead log, and is emptied rather than recovered after a crash, which suits a cache. Embeddings are stored as `real[]` rather than `vector`, because they are only read back by key.
- **Bounded without a background task.** Entries expire `EMBEDDING_SHARED_CACHE_TTL_SECONDS` after they are written (one day by default) and are ignored when expired. Every 1,000 writes, a process deletes expired rows and then the oldest rows beyond `EMBEDDING_SHARED_CACHE_MAX_ROWS` (100,000 by default). The two deletes are separate statements: in one statement both would see the same snapshot, and the count would include rows the first delete removes.
- **An optimisation only.** Any database error, including a missing table, a pool timeout or a closed pool, is logged at most once a minute and treated as a miss, and the query is embedded by the provider.
- **Rows with the wrong number of dimensions are misses**, so a model configured with different dimensions under the same name cannot return unusable vectors.

## Measurements

`scripts/bench_query_cache.py` runs 4 processes, each with its own embedder and an in-memory cache of the default 2,048 entries, as API workers have. Each process embeds 2,000 queries drawn from a Zipf distribution (exponent 1.0) over 5,000 distinct queries, 8 at a time, with a different random seed per process. A stub provider answers after 150 ms. Two runs gave:

| Shared cache | Provider calls | Hit rate | Mean ms | p95 ms | p99 ms | Wall time s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no | 3,182 / 3,183 | 60.2% | 63.6 / 64.9 | 161 | 165 | 16.7 / 17.1 |
| yes | 2,027 / 2,033 | 74.7% / 74.6% | 41.7 / 42.4 | 166 / 168 | 174 / 178 | 11.2 / 11.4 |

The shared cache removed about 36% of provider calls and a third of the mean embedding time. The cost is on misses, which now read PostgreSQL before calling the provider: p95 rose by 5 to 7 ms and p99 by 9 to 14 ms.

This measures the step the cache changes, query embedding, with a stub provider of fixed latency. It is not an end-to-end load test through the HTTP API with a real model; retrieval and fusion are unchanged by this decision.

## Consequences

- Popular queries are embedded once for all processes, and a restarted process finds them already cached.
- No new infrastructure: the cache lives in the existing database and is created by a migration.
- Each in-memory miss costs one indexed read in PostgreSQL, and each provider call one write, on the same connection pool as searches.
- The cache is off by default. Turning it on only pays when several processes, or restarts, see the same queries.
- Entries are not refreshed on use, so a popular query is re-embedded once per TTL.
