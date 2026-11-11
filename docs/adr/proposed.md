# Proposed decisions (open)

These decisions are not final. When one is made, record it as an ADR and update the index in
[README.md](README.md). Last updated: 2026-11-11.

| Decision | Current leaning | Still to settle | Status |
|---|---|---|---|
| LLM provider and model | Google Gemini via Google AI Studio (free tier) | Exact model name (pick from the free models AI Studio lists); confirm real rate limits; privacy terms (free-tier data may be used for training, so use only public or sample documents) | Leaning |
| Vector store | Chroma (local, no server) | Confirm it works well on Windows | Leaning |
| Embedding model | Not chosen | Local model (free, private) versus Gemini embeddings; check install size on Windows. Changing it later requires re-indexing. | Open |
| Supported document types | TXT and MD now; PDF and DOCX in Phase 1 | OCR for scanned files is a possible extension | Leaning |
| User interface | CLI now; web UI in Phase 4 | Streamlit versus Gradio | Open |
| Evaluation approach | Ragas-style metrics on our own question set (Phase 3) | Confirm with the supervisor whether measured results are required | Leaning |
| Fallback LLM provider | Groq (no training on data, faster) | Only if Gemini limits or privacy become a problem | Open |
