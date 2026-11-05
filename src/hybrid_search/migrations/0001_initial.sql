create extension if not exists vector;

create table documents (
    id           text primary key,
    title        text not null default '',
    content_hash text not null,
    metadata     jsonb not null default '{}',
    created_at   timestamptz not null default now(),
    updated_at   timestamptz not null default now()
);

-- One row per chunk. The document title is stored with each chunk so that it
-- contributes to both keyword and vector matching for every chunk.
create table chunks (
    id            bigint generated always as identity primary key,
    document_id   text not null references documents (id) on delete cascade,
    ordinal       integer not null,
    heading       text not null default '',
    content       text not null,
    token_count   integer not null,
    -- sha256 of the model name and the exact text that was embedded. Chunks with an
    -- identical key reuse the stored embedding instead of calling the model again.
    embedding_key text not null,
    embedding     vector(768) not null,
    search_vector tsvector generated always as (
        setweight(to_tsvector('english', heading), 'A') || setweight(to_tsvector('english', content), 'B')
    ) stored,
    unique (document_id, ordinal)
);

create index chunks_embedding_hnsw on chunks using hnsw (embedding vector_cosine_ops) with (m = 16, ef_construction = 64);
create index chunks_search_vector_gin on chunks using gin (search_vector);
create index chunks_embedding_key on chunks (embedding_key);
