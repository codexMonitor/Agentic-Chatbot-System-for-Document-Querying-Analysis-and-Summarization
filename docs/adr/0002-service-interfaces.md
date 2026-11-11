# ADR-0002: Protocol-based service interfaces and a container

- **Status:** Accepted
- **Date:** 2026-11-11

## Context

The LLM and vector store provider are not decided yet and will change during the project. Tests must
run offline without API keys.

## Decision

Define the services as `typing.Protocol` interfaces in `services/base.py` (`LLMService`,
`VectorStore`). Nodes depend only on these interfaces (invariant I1). Only `services/container.py`
chooses concrete providers from `Settings` (I2). Each node is built by a factory
`make_<name>(llm, store, ...)` returning `run(state) -> dict` (I3). An offline `FakeLLM` and
`InMemoryVectorStore` are the default providers.

## Consequences

- Swapping providers means adding an adapter and registering it in the container; nodes do not change.
- All tests can run offline against the fake implementations.
- Interface changes (such as adding chunk metadata) touch every node and test, so they need care.
