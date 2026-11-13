import json
from pathlib import Path

from doc_agent.eval.runner import main


def test_rescore_adds_key_fact_metrics_without_loading_services(tmp_path, monkeypatch, capsys):
    results_path = tmp_path / "old-results.json"
    output_path = tmp_path / "rescored" / "results.json"
    results_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "metrics": {"intent_accuracy": 0.75},
                "results": [
                    {
                        "id": "05",
                        "answer": "The values are 106,1 MeV and 1,6 MeV.",
                        "retrieved_contexts": ["A measured spread of 1,6 MeV."],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    dataset_path = Path(__file__).parents[1] / "eval" / "dataset.jsonl"

    def no_llm(*args, **kwargs):
        raise AssertionError("rescore must not load settings or providers")

    monkeypatch.setattr("doc_agent.eval.runner.load_settings", no_llm)
    monkeypatch.setattr("doc_agent.eval.runner.build_services", no_llm)

    assert (
        main(
            [
                "rescore",
                "--results",
                str(results_path),
                "--dataset",
                str(dataset_path),
                "--out",
                str(output_path),
            ]
        )
        == 0
    )

    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert output["metrics"]["intent_accuracy"] == 0.75
    assert output["metrics"]["key_fact_answer_rate"] == 1
    assert output["metrics"]["key_fact_context_rate"] == 0.5
    assert output["results"][0]["key_facts"] == ["106.1", "1.6"]
    assert output["results"][0]["missing_key_facts"] == {
        "answer": [],
        "context": ["106.1"],
    }
    assert "without LLM calls" in capsys.readouterr().out
