# Known issues

| ID | Title | Status | Planned fix |
|---|---|---|---|
| [KI-001](#ki-001-sources-shown-for-unrelated-questions) | Sources shown for unrelated questions | Open | Phase 3 |
| [KI-002](#ki-002-follow-up-questions-do-not-use-earlier-turns) | Follow-up questions do not use earlier turns | Open | Not scheduled |

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

## KI-002: Follow-up questions do not use earlier turns

Since Phase 2 the conversation is saved per `--thread` and restored after a restart, but the router and
the nodes build their prompts from the latest user message only. A follow-up such as "and who wrote it?"
is therefore retrieved and answered without the earlier turns.

- **Where:** `last_user_text` in `src/doc_agent/nodes/_utils.py`, used by every node.
- **Why:** Phase 2 (T2.3) specified persistence only; using the history in prompts (for example
  rewriting a follow-up into a standalone question) was not specified and is not a new node's worth of
  work to guess at. See [ADR-0008](adr/0008-checkpointing.md).
- **Impact:** the Phase 2 manual check "follow-up questions use earlier turns after restarting with the
  same `--thread`" cannot pass yet: the history is restored (`messages` grows) but not used.
- **Planned fix:** not scheduled; needs a decision (query rewriting versus passing recent turns to the
  prompt) and an evaluation case in Phase 3.
