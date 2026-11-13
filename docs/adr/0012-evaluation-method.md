# ADR-0012: Evaluation method

- **Status:** Accepted
- **Date:** 2026-11-13

## Context

Phase 3 needs a reproducible question set and a local evaluation runner for routing, retrieval,
citations and latency. The phase procedure requested ADR-0009 for this decision, but ADR-0009 is
already accepted for the nodes' provider-neutral error helpers, and ADR-0010 and ADR-0011 are also
in use. The human chose ADR-0012 to preserve the existing decision history.

The seed file contains 15 draft question/answer pairs for a physics paper. Those answers have not
been verified against the PDF and must be reviewed by the human before use.

## Decision

Keep the seed rows in a JSONL evaluation dataset using their provided questions, expected answers,
source filename and intent; do not add answers that are not in the seed file. Run QA and analysis
questions by default and skip summarization questions unless `--include-summarize` is selected.
Record intent accuracy, retrieval hit rate, invalid citation rate, and average latency per question
and node, along with the run settings.

Keep LLM-judged metrics opt-in. When `--judge` is requested and Ragas plus a real Gemini judge are
available, use the Ragas collections API for faithfulness, answer relevance and context relevance.
Otherwise retain `null` scores and record the reason. The pipeline call estimate and count exclude
embedding requests and optional judge calls.

## Consequences

- Evaluation runs can be performed offline with `FakeLLM` and the in-memory store.
- Real evaluation and Ragas judging remain human-run; no API call is needed to develop or test the
  harness.
- The reference answers are drafts until the human verifies all 15 against the PDF.
- Changing `--k`, `--chunk-size`, and `--chunk-overlap` makes runs comparable without editing
  source code.
