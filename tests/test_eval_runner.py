import json
import os

from doc_agent.eval.runner import main


def _write_dataset(path):
    rows = [
        {
            "id": "qa-1",
            "question": "What are Scrum sprints?",
            "reference_answer": "Scrum organizes work into sprints.",
            "source_docs": ["notes.txt"],
            "expected_intent": "qa",
        },
        {
            "id": "analysis-1",
            "question": "Compare Scrum and Kanban.",
            "reference_answer": "The excerpt mentions both methods.",
            "source_docs": ["notes.txt"],
            "expected_intent": "analysis",
        },
        {
            "id": "summary-1",
            "question": "Summarize the document.",
            "reference_answer": "The document discusses Scrum and Kanban.",
            "source_docs": ["notes.txt"],
            "expected_intent": "summarize",
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _make_offline_inputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in [name for name in os.environ if name.startswith("DOC_AGENT_")]:
        monkeypatch.delenv(name)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.txt").write_text(
        "Scrum organizes work into sprints. Kanban organizes work into a flow.",
        encoding="utf-8",
    )
    dataset = tmp_path / "dataset.jsonl"
    _write_dataset(dataset)
    return dataset, docs


def test_runner_writes_offline_metrics_and_null_judge_scores(tmp_path, monkeypatch, capsys):
    dataset, docs = _make_offline_inputs(tmp_path, monkeypatch)
    output_path = tmp_path / "results" / "baseline.json"

    assert (
        main(
            [
                "--dataset",
                str(dataset),
                "--docs",
                str(docs),
                "--limit",
                "3",
                "--k",
                "2",
                "--chunk-size",
                "80",
                "--chunk-overlap",
                "10",
                "--out",
                str(output_path),
            ]
        )
        == 0
    )

    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert output["counts"] == {
        "dataset_rows": 3,
        "run": 2,
        "skipped": 1,
        "llm_calls_expected": 4,
        "llm_calls_actual": 4,
    }
    assert output["settings"] == {
        "chunk_size": 80,
        "chunk_overlap": 10,
        "k": 2,
        "provider": "fake",
        "model": "",
        "vector_store": "memory",
        "docs": str(docs),
    }
    assert output["metrics"]["intent_accuracy"] == 1
    assert output["metrics"]["retrieval_hit_rate"] == 1
    assert output["metrics"]["invalid_citation_rate"] == 0
    assert output["metrics"]["key_fact_answer_rate"] is None
    assert output["metrics"]["key_fact_context_rate"] is None
    assert isinstance(output["metrics"]["avg_latency_ms"]["per_question"], float)
    assert output["metrics"]["avg_latency_ms"]["per_node"]
    assert output["metrics"]["faithfulness"] is None
    assert output["metrics"]["answer_relevance"] is None
    assert output["metrics"]["context_relevance"] is None
    assert output["judge"]["status"] == "disabled"
    console = capsys.readouterr().out
    assert "Expected LLM calls: 4" in console
    assert "Actual LLM calls: 4" in console
    assert "Judge metrics: null" in console


def test_runner_can_include_summarize_questions_offline(tmp_path, monkeypatch):
    dataset, docs = _make_offline_inputs(tmp_path, monkeypatch)
    output_path = tmp_path / "results" / "with-summarize.json"

    main(
        [
            "--dataset",
            str(dataset),
            "--docs",
            str(docs),
            "--include-summarize",
            "--out",
            str(output_path),
        ]
    )

    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert output["counts"]["run"] == 3
    assert output["counts"]["skipped"] == 0
    assert output["counts"]["llm_calls_expected"] == 6
    assert output["counts"]["llm_calls_actual"] == 6
    assert output["metrics"]["retrieval_hit_rate"] == 1


def test_judge_flag_keeps_scores_null_when_only_fake_provider_is_configured(
    tmp_path, monkeypatch, capsys
):
    dataset, docs = _make_offline_inputs(tmp_path, monkeypatch)
    output_path = tmp_path / "results" / "judge.json"

    main(
        [
            "--dataset",
            str(dataset),
            "--docs",
            str(docs),
            "--limit",
            "1",
            "--judge",
            "--out",
            str(output_path),
        ]
    )

    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert output["metrics"]["faithfulness"] is None
    assert output["metrics"]["answer_relevance"] is None
    assert output["metrics"]["context_relevance"] is None
    assert output["judge"]["status"] == "unavailable"
    assert "DOC_AGENT_LLM_PROVIDER=gemini" in output["judge"]["reason"]
    assert "Judge metrics: null" in capsys.readouterr().out


def test_runner_lists_missing_key_facts_for_each_question(tmp_path, monkeypatch):
    dataset, docs = _make_offline_inputs(tmp_path, monkeypatch)
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]
    rows[0]["key_facts"] = ["Scrum", "missing fact"]
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output_path = tmp_path / "results" / "key-facts.json"

    main(
        [
            "--dataset",
            str(dataset),
            "--docs",
            str(docs),
            "--limit",
            "1",
            "--out",
            str(output_path),
        ]
    )

    output = json.loads(output_path.read_text(encoding="utf-8"))
    result = output["results"][0]
    assert result["key_facts"] == ["Scrum", "missing fact"]
    assert result["missing_key_facts"] == {
        "answer": ["missing fact"],
        "context": ["missing fact"],
    }
    assert output["metrics"]["key_fact_answer_rate"] == 0.5
    assert output["metrics"]["key_fact_context_rate"] == 0.5
