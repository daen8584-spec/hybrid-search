# 0008. Rerank the top hybrid results with a cross-encoder, fused by rank

- Status: Accepted
- Date: 2026-11-05

## Context

Both first-stage retrievers score documents without reading them together with the query: BM25 counts terms and the vector search compares embeddings computed separately for the query and the document. A cross-encoder reranker reads the query and a document as one input and scores their relevance directly, which is usually more accurate. It is also far slower, because every query and document pair is a separate model evaluation, so it can only be applied to a short list of candidates.

A reranker can also make results worse. A small model may judge relevance less well than a strong embedding model on a specialised domain, and replacing the first-stage order with its order would then discard a good ranking.

## Decision

- Add an optional second stage that sends the top `RERANK_DEPTH` documents (default 20) to a reranking endpoint in the common `/rerank` format, which llama.cpp's server, Jina and Cohere-compatible gateways implement. Each document is sent as its title and the chunk it was ranked by.
- Combine the reranker's order with the first-stage order by Reciprocal Rank Fusion, with the reranker weighted by `RERANK_WEIGHT` ([0003](0003-reciprocal-rank-fusion.md)). A weight of 0 keeps the first-stage order and an infinite weight uses the reranker's order. Documents beyond the rerank depth keep their first-stage order after the reranked ones.
- Choose the weight with `tune-rerank` on the train split, so the test queries are not used for tuning, and report the result on the test split with `eval --rerank`.
- Default to `jina-reranker-v1-turbo-en`, a 38M-parameter English cross-encoder small enough to run on the same machine as the rest of the service.
- Reranking is opt-in per request (`rerank=true`), because it adds a model call to every search.

## Consequences

- Search quality can improve where the first stage ranks relevant documents too low, and the weight bounds the harm where the reranker is weaker than the first stage.
- Reranked searches take the time of one reranker call over the top documents on top of the hybrid search; the evaluation reports that latency.
- The reranker is another service to run and monitor. Failures return 502 for reranked requests only.
