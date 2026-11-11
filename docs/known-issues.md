# Known issues

| ID | Title | Status | Planned fix |
|---|---|---|---|
| [KI-001](#ki-001-sources-shown-for-unrelated-questions) | Sources shown for unrelated questions | Open | Phase 3 |

## KI-001: Sources shown for unrelated questions

Chroma returns the nearest `k` chunks even for unrelated questions, so sources are printed next to a
"not in the documents" answer; the "nothing relevant found" path only triggers when the store is empty.

- **Where:** `ChromaVectorStore.search` in `src/doc_agent/services/vectorstore.py`, used by the QA node
  (`src/doc_agent/nodes/qa.py`) and printed by the CLI (`src/doc_agent/cli.py`). It does not happen with
  `InMemoryVectorStore`, which only returns chunks that share keywords with the question.
- **Why:** nearest-neighbour search has no relevance cutoff: it always returns the `k` closest vectors,
  however distant they are. See [ADR-0005](adr/0005-vector-store-and-embeddings.md).
- **Impact:** for a question that the documents cannot answer, the model usually replies that it is not
  in the documents (the QA prompt asks for this), but `Sources:` still lists `k` unrelated chunks, which
  can mislead the reader. The "I couldn't find anything relevant" reply is only reached on an empty store.
- **Planned fix:** Phase 3: add a relevance threshold or retrieval grading, and measure the change with
  the evaluation harness (with and without the fix). Not fixed in Phase 1.
