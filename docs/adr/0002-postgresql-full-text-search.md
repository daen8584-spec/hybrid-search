# 0002. Use PostgreSQL full-text search with OR semantics for keyword retrieval

- Status: Accepted
- Date: 2026-11-05

## Context

Dense retrieval misses queries that depend on exact terms: gene and drug names, abbreviations, numbers and rare words that an embedding model represents poorly. A lexical retriever covers these cases. The usual choice is BM25 in a search engine such as Elasticsearch or OpenSearch, or through a PostgreSQL extension such as ParadeDB.

The service already depends on PostgreSQL for documents and vectors. A separate search engine adds a second store to keep consistent with the first, and the BM25 extensions are not available on every managed PostgreSQL offering.

PostgreSQL's built-in full-text search has two limitations. `ts_rank_cd` is not BM25: it has no inverse document frequency term and normalizes for document length differently. And `plainto_tsquery` joins every query term with AND, so a natural-language query matches only documents that contain all of its terms. On SciFact, where queries are full sentences (scientific claims), AND semantics matched no documents for typical queries.

## Decision

Use PostgreSQL full-text search over a stored, generated `tsvector` column with a GIN index. The chunk heading (the document title) is weighted `A` and the body `B`.

Build the query with `plainto_tsquery`, which handles stemming, stop words and punctuation safely for arbitrary user input, and rewrite its `&` operators to `|`. Every term becomes optional, and `ts_rank_cd` rewards chunks that contain more of the terms and contain them closer together.

## Consequences

- Keyword search has no additional infrastructure and is transactionally consistent with the vectors: a document is searchable by both retrievers as soon as its transaction commits.
- Ranking quality is below a tuned BM25 implementation. The evaluation command reports keyword retrieval separately, so the gap is measured rather than assumed, and replacing the keyword retriever with BM25 (for example `pg_search`) only requires a new implementation of one query.
- OR queries match more rows than AND queries, and `ts_rank_cd` is computed for every match because it cannot be served from the index. At this corpus size the cost is small; at millions of chunks, keyword candidates would need to be limited before ranking.
