-- Query embeddings shared by every API process (docs/adr/0009).
--
-- Unlogged: this is a cache, so it skips the write-ahead log and is emptied after a crash
-- rather than recovered. Embeddings are stored as real[] because the cache is only read
-- back by key, never searched by similarity.

create unlogged table query_embedding_cache (
    model      text not null,
    -- SHA-256 of the exact text sent to the provider, including the query prefix.
    query_hash bytea not null,
    embedding  real[] not null,
    created_at timestamptz not null default now(),
    primary key (model, query_hash)
);

create index query_embedding_cache_created_at on query_embedding_cache (created_at);
