# ADR-0010: Follow-up question rewriting

- **Status:** Accepted
- **Date:** 2026-11-12

## Context

Conversations are saved per thread ([ADR-0008](0008-checkpointing.md)), but every node used only the
latest message (known issue KI-002). A follow-up such as "How long are they?" was routed and searched
with those literal words, so multi-turn conversation did not work even though the history was stored.

## Decision

A new first node, `rewrite` (`nodes/rewrite.py`), turns the latest message into a standalone question
and stores it in `state["standalone_question"]`. The graph is
`START -> rewrite -> router -> (qa | analysis | summarize) -> END`.

- With no earlier messages in the thread, or `DOC_AGENT_REWRITE_HISTORY_TURNS=0`, no LLM call is made
  and the last message is used as is (the field is always overwritten, so an old value never leaks into
  a later turn).
- Otherwise one LLM call (`REWRITE` prompt) sees the last `DOC_AGENT_REWRITE_HISTORY_TURNS` messages
  before the new one (default 4; each message cut to 600 characters) and returns one standalone
  question in the same language, unchanged if it was already standalone.
- On `LLMError`, or an empty reply, the original question is used and a warning is logged.
- `question_text(state)` in `nodes/_utils.py` returns the standalone question if present, else the last
  message. The router, `qa` and `analysis` use it. `summarize` always covers the whole document and
  does not read the question.
- `FakeLLM` answers the `rewrite` task with the text after `New question:`, unchanged.

Alternatives considered: pass the recent turns straight into every node's prompt (more tokens on every
call, and retrieval would still search with the ambiguous text), or keyword-only history matching
(not reliable in Vietnamese or English).

## Consequences

- Retrieval and routing see a self-contained question, so follow-ups work; the answer prompts stay small.
- Cost: first turns of a thread cost no extra call; each follow-up costs one extra, short LLM call (and
  counts against rate limits).
- The rewrite can be wrong (for example resolving "they" to the wrong subject). The original question
  stays in `messages`, and the rewritten one is saved in the state (`standalone_question`); `--verbose`
  logs whether the rewrite changed it.
- Only the last few messages are considered, so a reference to something said much earlier is not
  resolved.
