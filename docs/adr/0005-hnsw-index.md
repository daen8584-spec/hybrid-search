# 0005. Use an HNSW index and measure its recall against exact search

- Status: Accepted
- Date: 2026-11-05

## Context

Exact nearest-neighbour search compares the query with every vector, which is linear in the corpus size. pgvector offers two approximate indexes. IVFFlat partitions vectors into lists and needs representative data before the index is built. HNSW builds a navigable graph incrementally, has better recall at a given latency, and needs no training step, at the cost of slower builds and more memory.

Approximate search trades recall for speed, and the trade-off is controlled at query time by `hnsw.ef_search`. Choosing a value without measuring it either wastes latency or silently loses relevant results.

## Decision

- Use HNSW with cosine distance and pgvector's defaults for the build parameters (`m = 16`, `ef_construction = 64`).
- Set `hnsw.ef_search` per query in a transaction-local setting, with a default of 100, and enable `hnsw.iterative_scan = strict_order` (pgvector 0.8 and later) so that requests for more rows than `ef_search` continue scanning the graph while still returning rows in exact distance order.
- Provide an exact mode that disables index scans for the query, and an `ann-sweep` command that measures recall@k of the index against exact search, the resulting nDCG@10, and latency for a range of `ef_search` values.

## Consequences

- The default `ef_search` is a measured choice and can be revisited with one command when the corpus or hardware changes.
- At the size of the evaluation corpus, exact search is fast enough that the index mainly demonstrates the mechanism. The sweep shows the recall cost, which grows with the corpus and determines whether the index is still acceptable at a larger scale.
- Index builds slow down ingestion as the corpus grows. For a large initial load, building the index after inserting the data is faster.
