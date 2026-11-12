# ADR-0009: Nodes may import the provider-neutral error helpers

- **Status:** Accepted
- **Date:** 2026-11-12

## Context

Invariant I1 says files in `nodes/` import only `services/base.py`, `prompts.py`, `state.py` and
`config.py`. Phase 2 needs nodes to catch `LLMError` and answer with a friendly message instead of
crashing the graph (T2.5), and the rewrite and router nodes catch it to fall back. `LLMError` lives in
`services/errors.py`, which imports no provider SDK and no concrete provider, so the reason for I1
(nodes must not depend on a concrete provider) is not affected.

## Decision

Nodes may also import `doc_agent.services.errors`. The rest of I1 is unchanged: no concrete provider,
no provider SDK, and only `services/container.py` chooses providers. `AGENTS.md` states the same
invariant.

`LLMError` has a `retryable` flag (default `False`) and an optional `retry_after_s`. `call_with_retry`
(used by the provider adapters, see [ADR-0007](0007-summarization-strategy.md)) sets them and retries
an `LLMError` raised by its callable only when `retryable` is set. Authentication and configuration
errors are never retried.

## Consequences

- Nodes need no knowledge of any SDK's exceptions.
- Nodes must not call `call_with_retry` themselves: an earlier version wrapped the summarization calls
  in a second retry layer on top of the adapter's, which could cost `(max_retries + 1)^2` attempts for
  one call. T2.8 removed it, so there is one retry layer, in the adapters.
- `call_with_retry` resolves `time.sleep` at call time, so tests can patch it.
