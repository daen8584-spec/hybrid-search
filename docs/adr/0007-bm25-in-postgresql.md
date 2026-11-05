# 0007. Rank keyword results with BM25 over an inverted index in PostgreSQL

- Status: Accepted
- Date: 2026-11-05
- Supersedes: [0002](0002-postgresql-full-text-search.md)

## Context

The first evaluation on SciFact measured the design in [0002](0002-postgresql-full-text-search.md): `ts_rank_cd` over an OR query reached nDCG@10 0.399, against 0.665 reported for BM25 on the same dataset in the BEIR paper. It was also slow, with a median of 374 ms on the test device, because an OR query over common scientific vocabulary matches most of the corpus and every match is ranked.

The weak keyword ranking made hybrid search worse than vector search alone (0.602 against 0.700), so the keyword retriever was subtracting quality rather than adding the exact-term matches it exists for.

`ts_rank_cd` has no inverse document frequency: a match on "cell" counts as much as a match on "angiotensin". BM25 weights each term by how rare it is and saturates repeated occurrences, which is what makes it the standard lexical baseline.

## Decision

Implement BM25 (k1 = 1.2, b = 0.75) in SQL over an inverted index stored in PostgreSQL:

- `chunk_terms` holds one row per lexeme and chunk, with the term frequency, the chunk length and the document id. Its primary key `(lexeme, chunk_id)` includes the other columns, so both document frequency counts and scoring are index-only scans.
- Lexemes come from the existing `search_vector` column, so tokenization, stemming and stop words are PostgreSQL's English configuration, identical for documents and queries.
- Ingestion writes the postings in the same transaction as the chunks, and the foreign key removes them with their chunk.
- The corpus size and mean chunk length are cached by the search service for a minute, since they change slowly.
- Each query scores only candidate chunks that contain a selective term: one whose document frequency is at most 10% of the corpus, or one of the rarest terms until their postings reach 1,000 rows. Candidates are scored with every query term. On a synthetic corpus with a Zipf vocabulary, this produced the same top 100 as scoring every match, two to five times faster.

## Consequences

- Keyword retrieval has the same ranking function as common search engines without leaving PostgreSQL, and stays transactionally consistent with the vectors.
- The inverted index adds one row per distinct lexeme per chunk and makes ingestion write correspondingly more rows.
- Candidate pruning is an approximation. Its effect is measurable: setting the threshold to 1.0 scores every match.
- Document frequencies are counted at query time. That keeps them exact and maintenance free, at a cost that grows with the frequency of the query terms. At much larger scale, a statistics table refreshed in the background would replace the counts.
