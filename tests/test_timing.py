import logging
import re

import pytest

from doc_agent.config import Settings
from doc_agent.graph import build_graph, memory_checkpointer
from doc_agent.nodes._utils import timed
from doc_agent.services import Services
from doc_agent.services.llm import FakeLLM
from doc_agent.services.vectorstore import InMemoryVectorStore

TIMING_LINE = re.compile(r"node=(\w+) duration_ms=(\d+)")


def run_graph(question: str, caplog) -> tuple[dict, list[str]]:
    store = InMemoryVectorStore()
    store.add("notes.txt", ["Scrum organizes work into sprints.", "Chroma stores embeddings."])
    graph = build_graph(Services(llm=FakeLLM(), store=store), Settings(), memory_checkpointer())
    with caplog.at_level(logging.INFO):
        result = graph.invoke(
            {"messages": [("user", question)]}, {"configurable": {"thread_id": "t"}}
        )
    return result, [match.group(1) for match in TIMING_LINE.finditer(caplog.text)]


@pytest.mark.parametrize(
    ("question", "expected_nodes"),
    [
        ("What is Scrum?", ["rewrite", "router", "qa"]),
        ("Compare Scrum and Chroma", ["rewrite", "router", "analysis"]),
        ("Summarize the document", ["rewrite", "router", "summarize"]),
    ],
)
def test_one_timing_line_per_executed_node(caplog, question, expected_nodes):
    _, logged = run_graph(question, caplog)
    assert logged == expected_nodes


def test_timing_does_not_change_the_node_result(caplog):
    result, _ = run_graph("What is Scrum?", caplog)
    assert result["intent"] == "qa"
    assert "Scrum" in result["answer"]


def test_timed_passes_the_result_through_and_logs_even_on_failure(caplog):
    def boom(state):
        raise RuntimeError("x")

    assert timed("ok", lambda state: {"answer": "a"})({}) == {"answer": "a"}
    with caplog.at_level(logging.INFO), pytest.raises(RuntimeError):
        timed("bad", boom)({})
    assert "node=bad duration_ms=" in caplog.text
