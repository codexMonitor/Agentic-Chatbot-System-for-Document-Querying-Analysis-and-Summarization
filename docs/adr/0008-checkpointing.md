# ADR-0008: Checkpointing and conversation memory

- **Status:** Accepted
- **Date:** 2026-11-12

## Context

Phase 0 and 1 kept conversation state in a `MemorySaver`, so it disappeared when the CLI exited. The
graph already needed an allow-list so that `Chunk` objects in `context` survive a checkpoint round trip
without "unregistered type" warnings.

## Decision

- `build_graph(services, settings, checkpointer=None)` creates a `SqliteSaver` when no checkpointer is
  passed. The file is `DOC_AGENT_CHECKPOINT_DB`, or `<DOC_AGENT_PERSIST_DIR>/checkpoints.sqlite` when
  that is empty (the default), so it is already git-ignored. Dependency: `langgraph-checkpoint-sqlite`.
- `graph.py` has one serializer configuration (`_make_serde`, the `Chunk` allow-list) used by both
  `sqlite_checkpointer(path)` and `memory_checkpointer()`. Tests pass `memory_checkpointer()`
  explicitly, so they never touch the disk.
- The conversation id is the LangGraph `thread_id`. The CLI takes `--thread <id>` (default `cli`);
  reusing an id after a restart continues that conversation.
- The CLI closes the SQLite connection on exit.

## Consequences

- Conversations survive restarts and each thread is independent. To forget them, delete the file or use
  another thread id.
- The file stores questions, answers and retrieved chunk text in plain text on disk; it is git-ignored,
  but treat it like the documents themselves.
- Only the allow-listed `Chunk` type is restored from checkpoints; adding another custom type to the
  state requires adding it to `_make_serde`.
- The nodes build their prompts from a standalone version of the latest question, so the stored earlier
  turns are used through the rewrite step (T2.7, see [KI-002](../known-issues.md)).
