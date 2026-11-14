# ADR-0013: Supabase pgvector

## Status
Accepted.

## Context
The outline requires Supabase while local development continues to use Chroma. Gemini's default
embedding has 3072 dimensions; pgvector's full-precision approximate indexes support only 2000.

## Decision
Use the Supabase Python client from the server with a secret key, RLS-enabled tables, and SQL
functions for atomic source replacement and cosine search. Store the embedding provider and model
per collection and refuse mismatches. Use exact search with full-precision 3072-dimensional vectors.

## Consequences
The human runs the SQL setup. Secret keys bypass RLS, so this local single-user application must
not be publicly exposed. Exact search trades speed for recall at larger scales. Changing embedding
models requires clearing the collection and re-ingesting; chat-model changes do not.
