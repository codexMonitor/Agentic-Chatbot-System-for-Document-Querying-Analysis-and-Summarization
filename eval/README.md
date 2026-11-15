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

`dataset_excel.jsonl` is a separate three-question review set built from the generated
`docs\sample_financial_report.xlsx` workbook. It is intentionally not merged into the PDF dataset.
Verify its reference answers against that sample workbook before using it for evaluation.

Install the optional judge dependency with `pip install -e ".[eval]"`. `--judge` uses the Ragas
collections API and a configured real Gemini model; otherwise the judged metrics remain `null`.
