# ADR-0014: Per-request chat model selection

## Status
Accepted.

## Context
Users need to switch from a paid chat provider to Gemini without changing document embeddings.
The Anthropic account and its exact model name are not yet verified.

## Decision
Add Anthropic as a registry adapter using the existing retry layer. The API accepts a provider and
model per request and lists only configured providers (plus the offline fake when selected).
The UI shows the choice and records provider and model on assistant messages. Keep the embedding
provider independent. The unverified default can be overridden after checking the Console.

## Consequences
Each request builds a graph using the shared checkpointer when its model differs from the default.
Anthropic SDK retries are disabled so the existing retry budget remains the only retry layer.
Provider authentication and credit failures are reported without provider error text or keys.
