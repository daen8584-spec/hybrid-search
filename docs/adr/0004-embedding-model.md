# 0004. Use an OpenAI-compatible embedding API and fix the dimension in the schema

- Status: Accepted
- Date: 2026-11-05

## Context

The service needs an embedding model for documents and queries. Hosted APIs are simple to use but add cost per token and send document content to a third party. Self-hosted models avoid both but need a serving process. The OpenAI embeddings request format is supported by OpenAI, by most gateways (LiteLLM, OpenRouter) and by local servers such as llama.cpp's `llama-server`, so a client for that format works with all of them.

pgvector indexes require a fixed dimension on the column, and vectors from different models are not comparable even when their dimensions match.

## Decision

- Use a single client for OpenAI-compatible endpoints, configured by base URL, model, key and dimension. It batches document requests, retries rate limits, server errors and timeouts with exponential backoff, validates the number and size of returned vectors, and caches query embeddings in memory (LRU).
- Default to `nomic-embed-text-v1.5` (768 dimensions) served locally by `llama-server`. It is small enough to run on modest hardware, has a published retrieval benchmark record, and is trained with task prefixes (`search_query: `, `search_document: `), which are configurable and can be empty for models that do not use them.
- Declare `vector(768)` in the schema. The API and the command-line tools compare the configured dimension with the column at startup and refuse to run on a mismatch.
- Include the model name in the content hash of each document and in the key of each embedding, so a model change never mixes vectors from two models.

## Consequences

- Moving to another provider is a configuration change, provided the dimension matches.
- Moving to a model with a different dimension requires a migration that changes the column type and rebuilds the index, followed by re-ingestion. Re-ingestion happens automatically for every document, because the changed model name changes every content hash.
- The query cache is per process and lost on restart. It removes the embedding call for repeated queries, which the load test measures as the difference between its cold and warm passes.
