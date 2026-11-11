# Sprint 01: Ingestion and basic QA

## Goal

Ingest real PDF/DOCX/TXT documents and answer questions with source citations, using a real LLM
(Gemini free tier) and a real vector store (Chroma with Gemini embeddings).

## Planned

- [x] T1.1 Settings additions and `.env` loading
- [x] T1.2 Chunk with metadata (interface change)
- [x] T1.3 LLM adapter (registry, `GeminiLLM`, `LLMError`, real-test plumbing)
- [x] T1.4 Real vector store with embeddings (Chroma, re-index guard)
- [x] T1.5 Loaders (`.pdf`, `.docx`)
- [x] T1.6 QA citations
- [x] T1.7 CLI sources output
- [x] T1.8 Docs (README, ADR-0004 to ADR-0006)

## Done

All planned tasks are implemented on branch `phase-1/ingestion`. Decisions are recorded in
[ADR-0004](../adr/0004-llm-provider.md), [ADR-0005](../adr/0005-vector-store-and-embeddings.md) and
[ADR-0006](../adr/0006-chunking.md). Offline tests pass; the two `real` tests (Gemini chat and Gemini
embeddings) passed against the free tier.

Pending human actions: ingest one PDF and one DOCX, ask 5 questions and confirm the citations show file
and position (the Phase 1 manual demo); try 2-3 of your own public documents; save at least 10 real
questions with expected answers into `eval/seed_questions.md` for Phase 3 (Vietnamese on Vietnamese
documents, English on English, and mixed-language cases).

## Review

To be filled in after the sprint review.

## Retrospective

- What went well: The agent really looks for my specific doc.
- What to improve: I don't think of anything yet.
- Actions for the next sprint: It should be in CAPSTONE_PROCEDURES.md already.
