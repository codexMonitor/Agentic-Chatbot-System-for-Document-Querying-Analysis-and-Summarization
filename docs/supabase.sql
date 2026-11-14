-- Run this whole script in the Supabase SQL editor. It does not delete existing data.
-- gemini-embedding-001 returns 3072 dimensions by default.
create extension if not exists vector with schema extensions;

create table if not exists public.doc_agent_collections (
    collection text primary key,
    embedder text not null
);

create table if not exists public.doc_agent_chunks (
    collection text not null references public.doc_agent_collections(collection),
    source text not null,
    position integer not null check (position >= 0),
    text text not null,
    embedding extensions.vector(3072) not null,
    primary key (collection, source, position)
);

alter table public.doc_agent_collections enable row level security;
alter table public.doc_agent_chunks enable row level security;
revoke all on public.doc_agent_collections, public.doc_agent_chunks from anon, authenticated;
grant select, insert, update, delete on public.doc_agent_collections, public.doc_agent_chunks
    to service_role;

-- A full-precision vector index supports only 2000 dimensions. Search 3072-dimensional
-- embeddings exactly, without an approximate vector index.
create or replace function public.doc_agent_init_collection(
    collection_name text, configured_embedder text
) returns text language sql as $$
    insert into public.doc_agent_collections as c (collection, embedder)
    values (collection_name, configured_embedder)
    on conflict (collection) do update set collection = excluded.collection
    returning c.embedder;
$$;

create or replace function public.doc_agent_replace_source(
    collection_name text, configured_embedder text, source_name text, chunks jsonb
) returns void language plpgsql as $$
declare recorded_embedder text;
begin
    select embedder into recorded_embedder from public.doc_agent_collections
        where collection = collection_name for update;
    if recorded_embedder is null or recorded_embedder <> configured_embedder then
        raise exception 'Embedding model mismatch; re-ingest after clearing the collection';
    end if;
    delete from public.doc_agent_chunks
        where collection = collection_name and source = source_name;
    insert into public.doc_agent_chunks (collection, source, position, text, embedding)
    select collection_name, source_name, (ordinality - 1)::integer,
           item->>'text', (item->'embedding')::text::extensions.vector
    from jsonb_array_elements(chunks) with ordinality as entries(item, ordinality);
end;
$$;

create or replace function public.doc_agent_delete_source(
    collection_name text, source_name text
) returns integer language plpgsql as $$
declare removed integer;
begin
    delete from public.doc_agent_chunks
        where collection = collection_name and source = source_name;
    get diagnostics removed = row_count;
    return removed;
end;
$$;

create or replace function public.doc_agent_match_chunks(
    collection_name text, query_embedding extensions.vector, match_count integer
) returns table (source text, "position" integer, text text)
language sql stable as $$
    select c.source, c.position, c.text
    from public.doc_agent_chunks as c
    where c.collection = collection_name
    order by c.embedding operator(extensions.<=>) query_embedding, c.source, c.position
    limit greatest(match_count, 0);
$$;

revoke all on function public.doc_agent_init_collection(text, text) from public, anon, authenticated;
revoke all on function public.doc_agent_replace_source(text, text, text, jsonb)
    from public, anon, authenticated;
revoke all on function public.doc_agent_delete_source(text, text) from public, anon, authenticated;
revoke all on function public.doc_agent_match_chunks(text, extensions.vector, integer)
    from public, anon, authenticated;
grant execute on function public.doc_agent_init_collection(text, text) to service_role;
grant execute on function public.doc_agent_replace_source(text, text, text, jsonb) to service_role;
grant execute on function public.doc_agent_delete_source(text, text) to service_role;
grant execute on function public.doc_agent_match_chunks(text, extensions.vector, integer)
    to service_role;
