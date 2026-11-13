import json
from pathlib import Path

import pytest

from doc_agent.eval.runner import _read_dataset

VALID_INTENTS = {"qa", "analysis", "summarize"}
REQUIRED_FIELDS = {
    "id",
    "question",
    "reference_answer",
    "source_docs",
    "expected_intent",
}
OPTIONAL_FIELDS = {"key_facts"}


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
        assert REQUIRED_FIELDS <= set(row) <= REQUIRED_FIELDS | OPTIONAL_FIELDS
        assert isinstance(row["id"], str)
        assert isinstance(row["question"], str) and row["question"]
        assert isinstance(row["reference_answer"], str) and row["reference_answer"]
        assert isinstance(row["source_docs"], list)
        assert all(isinstance(source, str) and source for source in row["source_docs"])
        assert row["expected_intent"] in VALID_INTENTS
        if "key_facts" in row:
            assert isinstance(row["key_facts"], list)
            assert all(isinstance(fact, str) and fact.strip() for fact in row["key_facts"])


def test_dataset_validator_accepts_optional_key_facts(tmp_path):
    rows = [
        {
            "id": "with-facts",
            "question": "Question?",
            "reference_answer": "Answer.",
            "source_docs": ["paper.pdf"],
            "expected_intent": "qa",
            "key_facts": ["1.6"],
        },
        {
            "id": "without-facts",
            "question": "Another question?",
            "reference_answer": "Another answer.",
            "source_docs": ["paper.pdf"],
            "expected_intent": "qa",
        },
    ]
    path = tmp_path / "dataset.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    assert _read_dataset(path) == rows


@pytest.mark.parametrize("key_facts", ["not-a-list", ["", 1]])
def test_dataset_validator_rejects_invalid_key_facts(tmp_path, key_facts):
    row = {
        "id": "invalid",
        "question": "Question?",
        "reference_answer": "Answer.",
        "source_docs": ["paper.pdf"],
        "expected_intent": "qa",
        "key_facts": key_facts,
    }
    path = tmp_path / "dataset.jsonl"
    path.write_text(json.dumps(row), encoding="utf-8")

    with pytest.raises(ValueError, match="key_facts must be a list"):
        _read_dataset(path)
