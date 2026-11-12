# ADR-0007: Summarization strategy

- **Status:** Accepted
- **Date:** 2026-11-12

## Context

The Phase 1 summarizer read only the first 20 chunks, so long documents were silently truncated, and it
called the LLM one chunk after another. A document can also be longer than the model's context window,
so "paste everything into one prompt" does not work in general. The free Gemini tier allows only about
15 requests per minute per model, so the strategy must be gentle with request rates.

## Decision

Hierarchical (map-reduce) summarization in `nodes/summarize.py`, with no chunk cap:

1. **Map:** summarize every chunk (`SUMMARIZE_CHUNK`) on a `ThreadPoolExecutor` of
   `DOC_AGENT_SUMMARIZE_WORKERS` threads (default 2).
2. **Reduce:** while more than one summary remains, merge consecutive groups of
   `DOC_AGENT_SUMMARIZE_GROUP_SIZE` summaries (default 8) with `SUMMARIZE_REDUCE`; a leftover group of
   one is carried up without an LLM call.
3. Return the single remaining summary.

Order is preserved at every level, so the summary follows the document order. The defaults are low on
purpose for the free tier; with a paid provider both can be raised. Every LLM call goes through
`call_with_retry` (see [ADR-0009](0009-nodes-may-use-error-helpers.md)), so a rate-limit error in one
worker is retried instead of aborting the whole summary. If a call still fails, pending work is
cancelled and the node returns the friendly "language model is unavailable" reply.

## Consequences

- No truncation: `N` chunks cost `N` map calls plus about `N / (g - 1)` merge calls for group size `g`
  (100 chunks with `g = 8` is 100 + 16 = 116 calls).
- Wall-clock time is bounded by the provider's request rate, not by the worker count. At 15 requests per
  minute, 116 calls take at least 8 minutes. A real run with the default `DOC_AGENT_LLM_MAX_RETRIES=3`
  failed on the free tier because the backoff (1, 2 and 4 seconds) is shorter than the rate-limit window;
  set `DOC_AGENT_LLM_MAX_RETRIES=8` there (see the Phase 2 sprint note).
- Chunk summaries lose cross-chunk context, and every merge level can drop detail. Context-aware
  hierarchical merging (arXiv 2502.00977) is a possible later improvement, to be judged with the
  evaluation harness in Phase 3.
- Summaries still have no per-chunk citations (`sources` stays empty).
