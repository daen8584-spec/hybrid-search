# Architecture Decision Records

Each record describes a significant design decision, the context that led to it and its consequences, following the format described in [0001](0001-record-architecture-decisions.md).

| Number | Decision |
| --- | --- |
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions |
| [0002](0002-postgresql-full-text-search.md) | Use PostgreSQL full-text search with OR semantics for keyword retrieval (superseded by 0007) |
| [0003](0003-reciprocal-rank-fusion.md) | Combine keyword and vector results with Reciprocal Rank Fusion |
| [0004](0004-embedding-model.md) | Use an OpenAI-compatible embedding API and fix the dimension in the schema |
| [0005](0005-hnsw-index.md) | Use an HNSW index and measure its recall against exact search |
| [0006](0006-synchronous-ingestion.md) | Ingest synchronously and reuse embeddings by content hash |
| [0007](0007-bm25-in-postgresql.md) | Rank keyword results with BM25 over an inverted index in PostgreSQL |
