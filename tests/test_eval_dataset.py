import json
from pathlib import Path

VALID_INTENTS = {"qa", "analysis", "summarize"}
REQUIRED_FIELDS = {
    "id",
    "question",
    "reference_answer",
    "source_docs",
    "expected_intent",
}


def test_evaluation_dataset_rows_match_the_schema():
    dataset_path = Path(__file__).parents[1] / "eval" / "dataset.jsonl"
    rows = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 15
    assert len({row["id"] for row in rows}) == len(rows)
    for row in rows:
        assert set(row) == REQUIRED_FIELDS
        assert isinstance(row["id"], str)
        assert isinstance(row["question"], str) and row["question"]
        assert isinstance(row["reference_answer"], str) and row["reference_answer"]
        assert isinstance(row["source_docs"], list)
        assert all(isinstance(source, str) and source for source in row["source_docs"])
        assert row["expected_intent"] in VALID_INTENTS
