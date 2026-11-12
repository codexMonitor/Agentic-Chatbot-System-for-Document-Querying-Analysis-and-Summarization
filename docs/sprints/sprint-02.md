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

## Done

All planned tasks are implemented on branch `phase-2/agentic-routing`, one or more commits per task.
Decisions: [ADR-0007](../adr/0007-summarization-strategy.md),
[ADR-0008](../adr/0008-checkpointing.md) and
[ADR-0009](../adr/0009-nodes-may-use-error-helpers.md) (a deviation from invariant I1).

Measurements:

| Measurement | Result |
|---|---|
| Router accuracy, `FakeLLM` (offline, keyword rules, 20 examples) | 0.75 (15/20) |
| Router accuracy, real `gemini-3.5-flash-lite` (20 examples) | 1.00 (20/20) |
| Summarize 100 chunks (97 from one PDF, 3 from `sample.txt`; 116 LLM calls), Gemini free tier, 2 workers, `DOC_AGENT_LLM_MAX_RETRIES=8` | 383.7 s |

The same run with the default `DOC_AGENT_LLM_MAX_RETRIES=3` failed after 250 s on the 15 requests per
minute free-tier limit (the chatbot returned the friendly "unavailable" reply instead of crashing).

Adapted to the existing code: `call_with_retry` was extended (a `retryable` flag on `LLMError`) instead
of adding a second retry mechanism; the `Chunk` serializer setting is shared by the SQLite and in-memory
checkpointers; `sources` is reset to `[]` by every error path; the existing CLI UTF-8 handling is kept.

Pending human actions: review `tests/data/intents.jsonl` (drafted by the agent); run a document longer
than the model context through summarize and judge coherence; try `--thread` after a restart. Note
[KI-002](../known-issues.md): follow-ups are saved and restored but the nodes do not use earlier turns
yet, so that part of the manual check cannot pass.

## Review

To be filled in after the sprint review.

## Retrospective

- What went well:
- What to improve:
- Actions for the next sprint:
