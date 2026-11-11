# Proposed decisions (open)

These decisions are not all final. When one is made, record it as an ADR and update the index in
[README.md](README.md). Decided rows are kept for reference. Last updated: 2026-11-11 (Phase 1).

| Decision | Current leaning | Still to settle | Status |
|---|---|---|---|
| LLM provider and model | Google Gemini (AI Studio free tier), `gemini-3.5-flash-lite` | Nothing for now. Free-tier data may be used for training, so use only public or sample documents; a paid provider is added later. | Decided ([ADR-0004](0004-llm-provider.md)) |
| Vector store | Chroma (local, persistent) | Nothing for now. | Decided ([ADR-0005](0005-vector-store-and-embeddings.md)) |
| Embedding model | Gemini `gemini-embedding-001` (multilingual) | `gemini-embedding-2` may be compared in Phase 3. Changing the embedding model requires re-ingesting. | Decided ([ADR-0005](0005-vector-store-and-embeddings.md)) |
| Supported document types | TXT, MD, PDF and DOCX | OCR for scanned files stays an extension idea. | Decided ([ADR-0006](0006-chunking.md) covers loading and chunking) |
| User interface | CLI now; web UI in Phase 4 | Leaning Streamlit; decide in Phase 4 (Streamlit versus Gradio). | Leaning |
| Evaluation approach | Ragas-style metrics on our own question set (Phase 3) | Confirm with the supervisor whether measured results are required | Leaning |
| Fallback LLM provider | Groq (no training on data, faster) | Only if Gemini limits or privacy become a problem | Open |
