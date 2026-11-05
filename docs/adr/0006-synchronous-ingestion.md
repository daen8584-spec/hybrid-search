# 0006. Ingest synchronously and reuse embeddings by content hash

- Status: Accepted
- Date: 2026-11-05

## Context

Embedding a document takes from milliseconds with a hosted API to seconds with a local model on modest hardware. Ingestion could run in a background queue, which keeps write requests fast but makes a document invisible to search for an unknown time and requires job tracking. Updates to an existing document usually change only part of its text, and re-embedding every chunk wastes time and, with a hosted API, money.

## Decision

- `PUT /documents/{id}` chunks, embeds and stores the document before responding. The response states whether the document was created, updated or unchanged, and how many chunks were embedded or reused.
- Each document stores a hash of its title, text, metadata, model and chunking settings. A request with the same hash returns `unchanged` without embedding or writing.
- Each chunk stores the SHA-256 of the model name and the exact text that was embedded. Before calling the model, ingestion looks up existing chunks with the same key and reuses their vectors.
- Embeddings are computed before the write transaction starts, so a slow model call never holds locks. The write itself (upsert of the document, replacement of its chunks) is a single transaction serialized per document by a transaction-scoped advisory lock, so concurrent writes to one document cannot interleave their chunks.

## Consequences

- A successful response means the document is searchable. Callers do not need to poll.
- Requests for large documents with a slow model take as long as the embedding takes. Callers with large batches should use concurrency, as the `ingest-beir` command does, or a background queue in front of the API if request latency matters more than read-after-write consistency.
- Editing one paragraph re-embeds only the chunks whose text changed. With overlapping chunks, the neighbours of a changed chunk can change too, which is correct, because their embedded text did change.
- When two writes to the same document race, the last transaction to commit wins, and the stored chunks always come from a single version.
