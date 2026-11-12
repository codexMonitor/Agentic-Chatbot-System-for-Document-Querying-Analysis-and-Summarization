# ADR-0009: Nodes may import the provider-neutral error helpers

- **Status:** Accepted
- **Date:** 2026-11-12

## Context

Invariant I1 says files in `nodes/` import only `services/base.py`, `prompts.py`, `state.py` and
`config.py`. Phase 2 needs nodes to do two things that live in `services/errors.py`:

- catch `LLMError` and answer with a friendly message instead of crashing the graph (T2.5);
- reuse `call_with_retry` for the parallel summarization calls (T2.2), as the spec forbids a second
  retry mechanism.

`services/errors.py` imports no provider SDK and no concrete provider, so the reason for I1 (nodes must
not depend on a concrete provider) is not affected.

## Decision

Nodes may also import `doc_agent.services.errors`. The rest of I1 is unchanged: no concrete provider,
no provider SDK, and only `services/container.py` chooses providers.

`LLMError` gets a `retryable` flag (default `False`). `call_with_retry` sets it on errors that were
transient (rate limit, temporary server error) and retries an `LLMError` raised by its callable only
when that flag is set. Authentication and configuration errors are never retried.

## Consequences

- One retry mechanism for adapters and nodes; nodes need no knowledge of any SDK's exceptions.
- Summarization wraps each call in `call_with_retry` on top of the retries inside the adapter, so a
  persistent rate limit can cost up to `(max_retries + 1)^2` attempts for one call. This is accepted
  because losing dozens of finished chunk summaries to one late failure is worse than a longer wait.
- `call_with_retry` now resolves `time.sleep` at call time, so tests can patch it.
