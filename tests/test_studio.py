import importlib


def test_studio_graph_imports_offline_without_application_checkpointer(monkeypatch, tmp_path):
    monkeypatch.setenv("DOC_AGENT_LLM_PROVIDER", "fake")
    monkeypatch.setenv("DOC_AGENT_VECTOR_STORE", "memory")
    monkeypatch.setenv("DOC_AGENT_EMBEDDING_PROVIDER", "fake")
    monkeypatch.setenv("DOC_AGENT_PERSIST_DIR", str(tmp_path / "store"))

    studio = importlib.import_module("doc_agent.studio")

    assert studio.graph.checkpointer is None
    result = studio.graph.invoke(
        {"messages": [("user", "Hello!")]}, {"configurable": {"thread_id": "studio-test"}}
    )
    assert result["intent"] == "chat"
