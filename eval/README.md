# Evaluation dataset

`dataset.jsonl` contains the 15 rows from `seed_questions.md`, converted to the runner's JSONL
schema. The reference answers are copied from the seed file; no additional answers were drafted.

**Human review required:** verify every `reference_answer` against the source PDF before using the
dataset for evaluation. The seed file itself notes that its answers were drafted from pages 1–10
and need checking. Add any additional questions or answers only after the human supplies them.

The `source_docs` entries use the PDF basename expected from ingestion. Place
`2611.13436v1.pdf` in `data\` for the documented run command, or pass the directory containing the
PDF to `--docs`. Questions marked with `expected_intent: "summarize"` remain in the dataset but
are skipped by default because summarization can be expensive.
