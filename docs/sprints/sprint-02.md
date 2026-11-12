# Sprint 02: Agentic routing, summarization, memory

## Goal

Reliable LLM routing, summarization that handles long documents, persistent conversations, timing
and error handling.

## Planned

- [x] T2.1 Router quality (examples in the prompt, tolerant parsing, `tests/data/intents.jsonl`)
- [x] T2.2 Hierarchical summarization (no chunk cap, parallel map, grouped reduce)
- [x] T2.3 Persistent memory (SQLite checkpointer, `--thread`)
- [x] T2.4 Timing logs (`node=<name> duration_ms=<n>`)
- [x] T2.5 Error handling (retryable-only retries, friendly reply, key masking)
- [x] T2.6 Docs (ADR-0007, ADR-0008, README, this note)
- [x] T2.7 Follow-up question rewriting (`rewrite` node, `standalone_question`; resolves KI-002)
- [x] T2.8 One retry layer that honors the server's retry delay (`retry_after_s`, jitter, default 8)
- [x] T2.9 Housekeeping (`AGENTS.md` invariant, ADR-0010, doc updates)

## Done

All planned tasks are implemented on branch `phase-2/agentic-routing`, one or more commits per task.
Decisions: [ADR-0007](../adr/0007-summarization-strategy.md),
[ADR-0008](../adr/0008-checkpointing.md),
[ADR-0009](../adr/0009-nodes-may-use-error-helpers.md) (a deviation from invariant I1) and
[ADR-0010](../adr/0010-follow-up-rewriting.md).

Measurements:

| Measurement | Result |
|---|---|
| Router accuracy, `FakeLLM` (offline, keyword rules, 20 examples) | 0.75 (15/20) |
| Router accuracy, real `gemini-3.5-flash-lite` (20 examples) | 1.00 (20/20) |
| Summarize 100 chunks (97 from one PDF, 3 from `sample.txt`; 116 LLM calls), Gemini free tier, 2 workers, `DOC_AGENT_LLM_MAX_RETRIES=8`, before T2.8 | 383.7 s |

The same run with `DOC_AGENT_LLM_MAX_RETRIES=3` failed after 250 s on the 15 requests per minute
free-tier limit (the chatbot returned the friendly "unavailable" reply instead of crashing). That run also
had a second retry layer in the summarize node, so one persistent rate limit could cost
`(retries + 1)^2` attempts. T2.8 removed that layer: the provider adapters are the only place that
retries, a 429 waits for the server's retry delay (at most 60 s, plus a little jitter), and the default
is now 8 retries. This is verified offline by a simulated 15-requests-per-minute provider (45 chunks,
default settings, virtual clock); no second long real summarization was run, so the 383.7 s figure is
from before the change.

T2.7: a follow-up such as "How long are they?" is rewritten into a standalone question before routing
and retrieval. Checked offline with a scripted LLM ("What is Scrum?" then "How long are they?" becomes
"How long are Scrum sprints?" and `qa` searches for that, including after rebuilding the graph on the
same SQLite file) and with one real Gemini call, which returned "How long are Scrum sprints?".

Adapted to the existing code: `call_with_retry` was extended (`retryable` flag and `retry_after_s` on
`LLMError`) instead of adding a second retry mechanism; the `Chunk` serializer setting is shared by the
SQLite and in-memory checkpointers; `sources` is reset to `[]` by every error path; the existing CLI
UTF-8 handling is kept.

Pending human actions: review `tests/data/intents.jsonl` (drafted by the agent; replace several examples
with harder, more realistic questions, since some overlap with the router prompt's style); run a document
longer than the model context through summarize and judge coherence; try follow-up questions after
restarting the CLI with the same `--thread` (should now pass, see KI-002). If your local `.env` still
sets `DOC_AGENT_LLM_MAX_RETRIES=3`, change it to 8 (a real environment variable or `.env` value
overrides the new default).

## Phase 2B: Plan alignment (continued)

### Done

- [x] T2B.1 Query-check chat route for greetings, thanks, and assistant meta questions; document
  questions continue to route to QA, analysis, or summarization.
- [x] T2B.2 Opt-in LangSmith tracing for the LangGraph run and Gemini chat/embedding calls;
  tracing is forced off in tests.
- [x] T2B.3 LangGraph Studio configuration, offline graph export, and `docs/graph.mmd`.
- [x] T2B.4 Deterministic simulated-rate-limit summarization test and documentation updates.

Measurements:

| Measurement | Result |
|---|---|
| Router accuracy, `FakeLLM` (offline, 26 examples) | 0.81 |
| Router accuracy, real `gemini-3.5-flash-lite` (26 examples, one run) | 0.96 |
| Studio environment used for validation | Python 3.12.7; `langgraph-cli` 0.4.42 |
| Traced sample question | `What is Scrum?` returned a cited answer from `data/sample.txt` |
| Deterministic rate-limit test | 10/10 consecutive runs passed with logical time and fixed jitter |

The six new chat intent examples are agent-drafted and await human review. The LangGraph CLI
requires Python 3.11+, while the application continues to support Python 3.10+. The human should
still run `langgraph dev` and capture the requested Studio screenshot.

## Review

To be filled in after the sprint review.

## Retrospective

- What went well:
- What to improve:
- Actions for the next sprint:
