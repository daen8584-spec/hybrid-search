-- BM25 keyword retrieval (docs/adr/0007).
--
-- chunk_terms is an inverted index: one row per (lexeme, chunk) with the term frequency
-- and the chunk length that BM25 needs. Its primary key covers every column the ranking
-- query reads, so scoring is served by index-only scans.

create function tsvector_term_count(v tsvector) returns integer
language sql immutable strict parallel safe
as $$ select coalesce(sum(greatest(cardinality(positions), 1)), 0)::integer from unnest(v) $$;

alter table chunks add column term_count integer generated always as (
    tsvector_term_count(
        setweight(to_tsvector('english', heading), 'A') || setweight(to_tsvector('english', content), 'B')
    )
) stored;

create table chunk_terms (
    lexeme       text not null,
    chunk_id     bigint not null references chunks (id) on delete cascade,
    document_id  text not null,
    tf           integer not null,
    chunk_length integer not null,
    primary key (lexeme, chunk_id) include (document_id, tf, chunk_length)
);

create index chunk_terms_chunk_id on chunk_terms (chunk_id);

insert into chunk_terms (lexeme, chunk_id, document_id, tf, chunk_length)
select t.lexeme, c.id, c.document_id, greatest(cardinality(t.positions), 1), c.term_count
from chunks c cross join lateral unnest(c.search_vector) as t;

-- Keyword search no longer evaluates tsquery against chunks.
drop index chunks_search_vector_gin;
