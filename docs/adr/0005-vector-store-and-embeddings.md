# ADR-0005: Vector store and embeddings

- **Status:** Accepted
- **Date:** 2026-11-11

## Context

Documents are Vietnamese and English, so retrieval needs a multilingual embedding model. The store must
persist between runs without a server, work on Windows, and stay independent of the chat LLM.

## Decision

- Embeddings: provider `gemini`, model `gemini-embedding-001` (multilingual; the human verified a
  Vietnamese/English similarity test). `gemini-embedding-2` is an alternative to compare in Phase 3.
  Embedders implement the `Embedder` Protocol and are registered in `EMBEDDERS`
  (`services/embeddings.py`). The Gemini embedder sends texts in batches of at most 100 and retries
  rate-limit errors with backoff. An offline hash-based `fake` embedder is used by tests.
- Vector store: Chroma (`chromadb`, installs from wheels on Windows) persisting under
  `DOC_AGENT_PERSIST_DIR` (default `.doc_agent_data`), using cosine distance. `source` and `position` are
  stored as metadata. The store receives an injected `Embedder` and knows no provider names. Stores are
  registered in `VECTOR_STORES` (`services/container.py`); `InMemoryVectorStore` remains for tests.
- Re-index guard: the embedder identity (`provider:model`) is recorded in the collection metadata. If the
  configured embedder differs at startup, an `IndexMismatchError` explains that the stored vectors are
  incompatible and documents must be re-ingested (or the persistence directory cleared). Vectors from
  different models are never mixed.
- Re-adding a file replaces its previous chunks.

## Consequences

- The LLM, the embedding model and the vector store can each change independently, except that
  changing the embedding model forces re-ingestion.
- Chroma always returns the nearest `k` chunks, so the "nothing relevant found" path of the QA node is
  only reached on an empty store; irrelevant questions rely on the prompt ("not in the documents").
  A relevance threshold is a Phase 2/3 candidate.
- Anonymous Chroma telemetry is disabled.
