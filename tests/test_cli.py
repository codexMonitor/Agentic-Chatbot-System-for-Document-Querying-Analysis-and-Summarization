import io
import sys

import pytest

from doc_agent import cli

ENV_VARS = [
    "DOC_AGENT_LLM_PROVIDER",
    "DOC_AGENT_LLM_MODEL",
    "DOC_AGENT_EMBEDDING_PROVIDER",
    "DOC_AGENT_EMBEDDING_MODEL",
    "DOC_AGENT_VECTOR_STORE",
    "DOC_AGENT_PERSIST_DIR",
    "DOC_AGENT_CHECKPOINT_DB",
]


@pytest.fixture(autouse=True)
def offline_env(monkeypatch, tmp_path):
    for name in ENV_VARS:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)


def run_cli(monkeypatch, capsys, argv, lines):
    answers = iter(lines)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    code = cli.main(argv)
    return code, capsys.readouterr().out


def make_doc(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Scrum organizes work into sprints.\n", encoding="utf-8")
    return path


def test_qa_answer_prints_sources(monkeypatch, capsys, tmp_path):
    code, out = run_cli(monkeypatch, capsys, [str(make_doc(tmp_path))], ["What is Scrum?", "quit"])
    assert code == 0
    assert "ingested notes.txt: 1 chunks" in out
    assert "[qa]" in out
    assert "Sources:\n[1] notes.txt#0" in out


def test_no_sources_block_when_nothing_found(monkeypatch, capsys, tmp_path):
    _, out = run_cli(monkeypatch, capsys, [str(make_doc(tmp_path))], ["xylophone", "quit"])
    assert "couldn't find" in out
    assert "Sources:" not in out


def test_skipped_file_message_stays(monkeypatch, capsys, tmp_path):
    _, out = run_cli(monkeypatch, capsys, [str(tmp_path / "missing.txt")], ["quit"])
    assert "! skipped" in out
    assert "file not found" in out


def test_startup_error_is_reported(monkeypatch, capsys):
    monkeypatch.setenv("DOC_AGENT_LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("DOC_AGENT_LLM_MODEL", "m")
    code, out = run_cli(monkeypatch, capsys, [], [])
    assert code == 1
    assert "GEMINI_API_KEY" in out


def test_vietnamese_output_survives_a_legacy_code_page(monkeypatch, tmp_path):
    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="cp1252"))
    path = tmp_path / "vi.txt"
    path.write_text("Scrum chia công việc thành các sprint.\n", encoding="utf-8")
    answers = iter(["Scrum chia công việc", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert cli.main([str(path)]) == 0
    sys.stdout.flush()
    assert "công việc" in raw.getvalue().decode("utf-8")


def test_chroma_persists_between_runs(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("DOC_AGENT_VECTOR_STORE", "chroma")
    monkeypatch.setenv("DOC_AGENT_PERSIST_DIR", str(tmp_path / "store"))
    run_cli(monkeypatch, capsys, [str(make_doc(tmp_path))], ["quit"])
    _, out = run_cli(monkeypatch, capsys, [], ["What is Scrum?", "quit"])
    assert "Sources:\n[1] notes.txt#0" in out
