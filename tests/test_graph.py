import pytest

from doc_agent.config import Settings
from doc_agent.graph import build_graph
from doc_agent.services import Services
from doc_agent.services.base import Chunk
from doc_agent.services.ingestion import split_text
from doc_agent.services.llm import FakeLLM
from doc_agent.services.vectorstore import InMemoryVectorStore

CHUNKS = [
    "LangGraph builds stateful agent workflows from nodes and conditional edges.",
    "Scrum organizes work into sprints with a sprint goal and a product backlog.",
    "Chroma is a local vector database for storing embeddings.",
]


@pytest.fixture
def graph():
    store = InMemoryVectorStore()
    store.add("notes.txt", CHUNKS)
    return build_graph(Services(llm=FakeLLM(), store=store), Settings())


def ask(graph, question, thread="t1"):
    return graph.invoke({"messages": [("user", question)]}, {"configurable": {"thread_id": thread}})


@pytest.mark.parametrize(
    ("question", "intent"),
    [
        ("What is Scrum?", "qa"),
        ("Summarize the document", "summarize"),
        ("Compare LangGraph and Scrum", "analysis"),
    ],
)
def test_router_intent(graph, question, intent):
    assert ask(graph, question)["intent"] == intent


def test_qa_returns_only_relevant_chunk(graph):
    result = ask(graph, "What is Scrum?")
    assert result["context"] == [Chunk(CHUNKS[1], "notes.txt", 1)]


def test_qa_without_match_says_nothing_found(graph):
    result = ask(graph, "xylophone")
    assert "couldn't find" in result["answer"]


def test_summarize_uses_all_chunks(graph):
    result = ask(graph, "Summarize the document")
    assert [chunk.text for chunk in result["context"]] == CHUNKS


def test_search_results_carry_source_and_position():
    store = InMemoryVectorStore()
    store.add("a.txt", ["alpha one", "beta two"])
    store.add("b.txt", ["gamma three", "beta four"])
    assert store.search("beta") == [Chunk("beta two", "a.txt", 1), Chunk("beta four", "b.txt", 1)]
    assert store.all_chunks()[2] == Chunk("gamma three", "b.txt", 0)


def test_adding_a_source_again_replaces_its_chunks():
    store = InMemoryVectorStore()
    store.add("a.txt", ["old one", "old two"])
    store.add("a.txt", ["new one"])
    assert store.all_chunks() == [Chunk("new one", "a.txt", 0)]


def test_summarize_empty_store():
    graph = build_graph(Services(llm=FakeLLM(), store=InMemoryVectorStore()), Settings())
    result = ask(graph, "Summarize the document")
    assert "No documents" in result["answer"]


def test_memory_persists_across_turns(graph):
    ask(graph, "What is Scrum?", thread="memory")
    result = ask(graph, "What is LangGraph?", thread="memory")
    assert len(result["messages"]) == 4


def test_split_text_overlap():
    chunks = split_text("x" * 1000, chunk_size=500, overlap=50)
    assert len(chunks) == 3
    assert all(len(chunk) <= 500 for chunk in chunks)
    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert current.startswith(previous[-50:])


def test_split_text_rejects_bad_overlap():
    with pytest.raises(ValueError):
        split_text("hello", chunk_size=10, overlap=10)
