# ADR-0001: Use LangGraph for orchestration

- **Status:** Accepted
- **Date:** 2026-11-11

## Context

The chatbot must route each user request to different behaviours (question answering, analysis,
summarization), keep conversation state across turns, and grow with new capabilities in later
phases (memory, error handling, evaluation).

## Decision

Use LangGraph (`StateGraph`) to orchestrate the agent. State is a typed `AgentState`; each capability
is a node; a router node plus conditional edges selects the path; a checkpointer (`MemorySaver` at
baseline) provides conversation memory per thread.

## Consequences

- Control flow is explicit and inspectable as a graph, which also supports generating diagrams.
- Conversation persistence is available through checkpointers without custom code.
- The project depends on LangGraph and its release cadence; node logic stays in plain Python
  functions so the dependency stays shallow.
