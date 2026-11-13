import pytest

from doc_agent.eval.metrics import calculate_key_fact_metrics


def test_key_fact_rates_normalize_case_and_decimal_comma_per_row():
    results = [
        {
            "id": "first",
            "key_facts": ["1.6", "Alpha"],
            "answer": "The value is 1,6 and alpha.",
            "retrieved_contexts": ["Alpha measured 1,6."],
        },
        {
            "id": "second",
            "key_facts": ["Beta", "Gamma", "Delta"],
            "answer": "BETA and gamma are in the answer.",
            "retrieved_contexts": ["Only beta appears here."],
        },
    ]

    metrics = calculate_key_fact_metrics(results)

    assert metrics["key_fact_answer_rate"] == pytest.approx(5 / 6)
    assert metrics["key_fact_context_rate"] == pytest.approx(2 / 3)
    assert results[0]["missing_key_facts"] == {"answer": [], "context": []}
    assert results[1]["missing_key_facts"] == {
        "answer": ["Delta"],
        "context": ["Gamma", "Delta"],
    }


def test_key_fact_metrics_are_null_when_no_rows_have_key_facts():
    results = [{"id": "empty", "answer": "answer", "retrieved_contexts": []}]

    assert calculate_key_fact_metrics(results) == {
        "key_fact_answer_rate": None,
        "key_fact_context_rate": None,
    }
    assert "missing_key_facts" not in results[0]
