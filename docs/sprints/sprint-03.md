# Sprint 03: Analysis and quality

## Goal

Add structured, cited analysis and an offline-capable evaluation runner for routing, retrieval,
citations and latency.

## Planned

- [x] T3.1 Structured analysis response with Summary, Key points, Evidence and Open questions
- [x] T3.2 Citation range validation and warning logs
- [x] T3.3 JSONL dataset drafted from all 15 seed rows
- [x] T3.4 Offline-capable evaluation runner, options and optional Ragas judge
- [ ] T3.5 Baseline (human-run against the real configuration)
- [ ] T3.6 Experiments (human-run; change one setting at a time)
- [x] T3.7 Evaluation docs and method decision ([ADR-0012](../adr/0012-evaluation-method.md);
  ADR-0009 was already in use)
- [ ] T3.8 LLM comparison (later phase prompt; human must choose the second provider)

## Done

T3.1–T3.4 and T3.7 are implemented. Offline tests exercise the analysis node, citation validator,
dataset schema, runner output and null judge metrics. No real provider or embedding calls were made
while implementing this work. The runner reports pipeline LLM invocation estimates and counts;
embedding and optional judge calls are separate and excluded from those numbers.

The dataset includes all 15 rows from `eval/seed_questions.md` without adding reference answers.
Every reference answer still needs human review against `2611.13436v1.pdf`. That PDF was not present
in the workspace when this sprint note was written; put it in `data\` or pass its containing
directory to `--docs` before running the baseline.

## Pending human actions

- Review every reference answer in `eval/dataset.jsonl` against the source PDF.
- Run the baseline before tuning, then run experiments and record the measured results in
  `docs/experiments.md`.
- Read at least 10 answers by hand and compare them with the metrics.
- Run `--judge` once on the selected final configuration if Ragas scoring is desired.

## Review

To be filled in after the sprint review.

## Retrospective

- What went well: Worked more effectively.
- What to improve: Reduce API calls for Copilot in Phase 4 (Making UI).
- Actions for the next sprint: Building UI with Streamlit.
