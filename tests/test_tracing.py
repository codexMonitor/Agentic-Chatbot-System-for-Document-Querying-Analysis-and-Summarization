import os
import socket

from conftest import force_langsmith_tracing_off
from doc_agent.config import Settings
from doc_agent.services.embeddings import GeminiEmbedder
from doc_agent.services.llm import GeminiLLM


class StubModels:
    def generate_content(self, *, model: str, contents: str):
        return type("Response", (), {"text": "offline answer"})()

    def embed_content(self, *, model: str, contents: list[str]):
        return type(
            "Response",
            (),
            {"embeddings": [type("Embedding", (), {"values": [0.1, 0.2]})() for _ in contents]},
        )()


class StubClient:
    def __init__(self) -> None:
        self.models = StubModels()


def test_test_configuration_overrides_enabled_tracing(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    force_langsmith_tracing_off(monkeypatch)

    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert "LANGSMITH_API_KEY" not in os.environ


def test_tracing_off_keeps_gemini_adapters_offline(monkeypatch):
    def reject_network(*args, **kwargs):
        raise AssertionError("Tracing unexpectedly attempted network access")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket, "create_connection", reject_network)

    llm = GeminiLLM(Settings(llm_model="test-model"), client=StubClient())
    embedder = GeminiEmbedder(Settings(embedding_model="test-embedding-model"), client=StubClient())

    assert llm.invoke("hello") == "offline answer"
    assert embedder.embed(["one", "two"]) == [[0.1, 0.2], [0.1, 0.2]]
