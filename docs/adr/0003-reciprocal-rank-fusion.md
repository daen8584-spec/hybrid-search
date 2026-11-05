# 0003. Combine keyword and vector results with Reciprocal Rank Fusion

- Status: Accepted
- Date: 2026-11-05

## Context

Hybrid search needs a way to merge two ranked lists. The scores are not comparable: `ts_rank_cd` is unbounded and depends on term frequency, while cosine similarity lies between -1 and 1 and its useful range depends on the embedding model. A weighted sum of scores requires normalizing both, and min-max normalization per query is sensitive to outliers and to the number of candidates.

## Decision

Use Reciprocal Rank Fusion (Cormack, Clarke and Buettcher, 2009). Each retriever returns its top 100 documents, and a document's fused score is the sum over retrievers of `weight / (k + rank)`, with `k = 60` and equal weights. Retrieval happens at chunk level, and each retriever collapses its chunks to one result per document (the best chunk) before fusion, so a document with many matching chunks does not take several positions.

The keyword query and the query embedding plus vector query run concurrently, so hybrid latency is close to the slower of the two rather than their sum.

## Consequences

- No score calibration or training data is needed, and fusion behaves the same whichever embedding model is used.
- Rank information alone discards the magnitude of a strong match. A learned combination or a cross-encoder reranker over the fused candidates can improve on RRF, at the cost of latency and, for a reranker, an additional model.
- `k` and the weights are parameters of `SearchService`. The evaluation command makes their effect measurable before a change.
