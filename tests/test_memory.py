import logging
import os

import pytest

from doc_agent import cli
from doc_agent.config import Settings
from doc_agent.graph import build_graph, sqlite_checkpointer
from doc_agent.services import Services
from doc_agent.services.base import Chunk
from doc_agent.services.llm import FakeLLM
from doc_agent.services.vectorstore import InMemoryVectorStore

CHUNKS = [
    "LangGraph builds stateful agent workflows from nodes and conditional edges.",
    "Scrum organizes work into sprints with a sprint goal and a product backlog.",
]


def make_services() -> Services:
    store = InMemoryVectorStore()
    store.add("notes.txt", CHUNKS)
    return Services(llm=FakeLLM(), store=store)


def thread(name: str) -> dict:
    return {"configurable": {"thread_id": name}}


def ask(graph, question: str, name: str = "t") -> dict:
    return graph.invoke({"messages": [("user", question)]}, thread(name))


def test_history_and_chunks_survive_a_new_graph_on_the_same_file(tmp_path, recwarn, caplog):
    db = tmp_path / "memory.sqlite"
    first_saver = sqlite_checkpointer(db)
    first = build_graph(make_services(), Settings(), first_saver)
    assert len(ask(first, "What is Scrum?")["messages"]) == 2
    first_saver.conn.close()

    second_saver = sqlite_checkpointer(db)
    second = build_graph(make_services(), Settings(), second_saver)
    with caplog.at_level(logging.WARNING):
        state = second.get_state(thread("t"))
    assert len(state.values["messages"]) == 2
    assert state.values["context"] == [Chunk(CHUNKS[1], "notes.txt", 1)]
    assert len(ask(second, "What is LangGraph?")["messages"]) == 4
    second_saver.conn.close()

    assert "unregistered" not in caplog.text
    assert not [w for w in recwarn if "unregistered" in str(w.message).lower()]


def test_threads_are_independent(tmp_path):
    saver = sqlite_checkpointer(tmp_path / "memory.sqlite")
    graph = build_graph(make_services(), Settings(), saver)
    ask(graph, "What is Scrum?", "a")
    assert len(ask(graph, "What is Scrum?", "b")["messages"]) == 2
    saver.conn.close()


def test_default_checkpoint_file_lives_under_persist_dir(tmp_path):
    settings = Settings(persist_dir=str(tmp_path / "data"))
    graph = build_graph(make_services(), settings)
    ask(graph, "What is Scrum?")
    graph.checkpointer.conn.close()
    assert (tmp_path / "data" / "checkpoints.sqlite").is_file()


def test_checkpoint_db_setting_overrides_the_default_location(tmp_path):
    settings = Settings(persist_dir=str(tmp_path / "data"), checkpoint_db=str(tmp_path / "c.db"))
    graph = build_graph(make_services(), settings)
    ask(graph, "What is Scrum?")
    graph.checkpointer.conn.close()
    assert (tmp_path / "c.db").is_file()
    assert not (tmp_path / "data").exists()


@pytest.fixture
def offline_cli(monkeypatch, tmp_path):
    # Run from an empty folder so the repo's real `.env` (Gemini, Chroma) is never loaded.
    monkeypatch.chdir(tmp_path)
    for name in [name for name in os.environ if name.startswith("DOC_AGENT_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("DOC_AGENT_PERSIST_DIR", str(tmp_path / "data"))
    doc = tmp_path / "notes.txt"
    doc.write_text("Scrum organizes work into sprints.\n", encoding="utf-8")

    def run(*argv: str, lines: list[str]) -> int:
        answers = iter(lines)
        monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
        return cli.main([str(doc), *argv])

    return run


def message_count(tmp_path, name: str) -> int:
    saver = sqlite_checkpointer(tmp_path / "data" / "checkpoints.sqlite")
    graph = build_graph(make_services(), Settings(), saver)
    state = graph.get_state(thread(name))
    saver.conn.close()
    return len(state.values.get("messages", []))


def test_cli_thread_option_continues_the_same_conversation(offline_cli, tmp_path):
    assert offline_cli("--thread", "alpha", lines=["What is Scrum?", "quit"]) == 0
    assert offline_cli("--thread", "alpha", lines=["What is Scrum?", "quit"]) == 0
    assert message_count(tmp_path, "alpha") == 4
    assert message_count(tmp_path, "cli") == 0


def test_cli_verbose_prints_timings_to_stderr(offline_cli, capsys):
    offline_cli("--verbose", lines=["What is Scrum?", "quit"])
    err = capsys.readouterr().err
    assert "node=router duration_ms=" in err
    assert "node=qa duration_ms=" in err


def test_cli_is_quiet_by_default(offline_cli, capsys):
    offline_cli(lines=["What is Scrum?", "quit"])
    assert "duration_ms" not in capsys.readouterr().err


def test_cli_default_thread_is_cli(offline_cli, tmp_path):
    offline_cli(lines=["What is Scrum?", "quit"])
    assert message_count(tmp_path, "cli") == 2
