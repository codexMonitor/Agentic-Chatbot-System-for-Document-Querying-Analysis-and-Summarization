# ADR-0011: LangSmith tracing and LangGraph Studio

- **Status:** Accepted
- **Date:** 2026-11-12

## Context

The approved project outline calls for LangSmith tracing for observability and LangGraph Studio
for graph development and debugging. Provider calls use the Google GenAI SDK directly, so they need
explicit LangSmith runs to appear alongside LangGraph's graph traces.

## Decision

Use the `langsmith` Python SDK's `traceable` decorator for Gemini chat and embedding calls, with
LLM and embedding run types and provider/model metadata. Mask configured API keys from trace inputs
and outputs. Keep tracing disabled by default with `LANGSMITH_TRACING=false`; the test suite also
forces tracing off and removes the LangSmith key. Studio runs the same graph through the local
LangGraph CLI without the application's SQLite checkpointer.

## Consequences

With tracing enabled, prompts and document text are sent to LangSmith, so only public or sample
documents should be used. The LangGraph CLI requires Python 3.11 or newer, while the application
continues to support Python 3.10; Studio therefore requires a Python 3.11+ environment.
