# ADR-0006: Chunking

- **Status:** Accepted
- **Date:** 2026-11-11

## Context

Answers must cite where information came from, so every chunk needs a stable identity. Documents are
Vietnamese and English, so the splitter must not assume ASCII or English sentence rules.

## Decision

Keep the fixed-size character window splitter (`split_text`: `DOC_AGENT_CHUNK_SIZE=500`,
`DOC_AGENT_CHUNK_OVERLAP=50`). Each chunk is a `Chunk(text, source, position)` where `source` is the file
name and `position` is the 0-based index of the chunk within that file; the vector store assigns
positions in `add(source, chunks)`. Prompts label context as `[n] (source#position) text`, and the QA
node returns `sources` as `{"n", "source", "position"}` for exactly the chunks given to the LLM.
Loaders extract plain text (PDF via `pypdf`, DOCX via `python-docx`, including table cells) before
splitting.

## Consequences

- Citations are reproducible: `file#position` identifies one chunk.
- Splitting ignores sentence and heading boundaries and may cut a sentence in two; structure-aware
  chunking is a possible later extension.
- Positions depend on the chunk size and overlap: changing them requires re-ingesting.
- Page numbers are not tracked.
