# ADR-0004: LLM provider (Gemini free tier for testing)

- **Status:** Accepted
- **Date:** 2026-11-11

## Context

Phase 1 needs a real LLM. The project is in testing and development, so cost must be zero, and the
documents are Vietnamese and English. A paid provider will be added later.

## Decision

Use Google Gemini through Google AI Studio (free tier) with the model `gemini-3.5-flash-lite`, which the
human verified on 2026-11-11. The adapter `GeminiLLM` uses the `google-genai` SDK; the key comes from the
`GEMINI_API_KEY` environment variable (loaded from `.env`) and is never stored in `Settings` or logged.
Providers are registered in `LLM_PROVIDERS` (`services/llm.py`) and selected by
`DOC_AGENT_LLM_PROVIDER`; the model name comes from `DOC_AGENT_LLM_MODEL`. All provider exceptions,
including rate limits, are wrapped in `LLMError`; rate-limit and server errors are retried with
exponential backoff up to `DOC_AGENT_LLM_MAX_RETRIES`, and requests time out after
`DOC_AGENT_LLM_TIMEOUT_S`.

## Consequences

- Switching to a paid provider (OpenAI, xAI, Groq) needs one adapter class, one registry entry, its SDK
  dependency and its key in `.env.example`; nodes, prompts and the graph do not change.
- Free-tier data may be used for training: use only public or sample documents.
- Free-tier rate limits apply; the retry/backoff handles short bursts only.
