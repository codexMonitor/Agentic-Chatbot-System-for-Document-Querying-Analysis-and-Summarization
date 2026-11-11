import pytest

from doc_agent.config import Settings
from doc_agent.services.base import Chunk
from doc_agent.services.container import _make_embedder, _make_store
from doc_agent.services.embeddings import FakeEmbedder, GeminiEmbedder
from doc_agent.services.errors import IndexMismatchError, LLMError
from doc_agent.services.vectorstore import ChromaVectorStore, InMemoryVectorStore

CORPUS = [
    "LangGraph builds stateful agent workflows from nodes and conditional edges.",
    "Scrum organizes work into sprints with a sprint goal and a product backlog.",
    "Chroma is a local vector database for storing embeddings.",
]


@pytest.fixture(params=["memory", "chroma"])
def store(request, tmp_path):
    if request.param == "memory":
        return InMemoryVectorStore()
    return ChromaVectorStore(str(tmp_path / "chroma"), FakeEmbedder(), "fake:")


def test_search_finds_the_relevant_chunk(store):
    store.add("notes.txt", CORPUS)
    top = store.search("sprint goal backlog", k=2)[0]
    assert top == Chunk(CORPUS[1], "notes.txt", 1)


def test_search_respects_k(store):
    store.add("notes.txt", CORPUS)
    assert len(store.search("sprint goal backlog graph vector", k=2)) <= 2


def test_search_on_empty_store(store):
    assert store.search("anything") == []
    assert store.all_chunks() == []


def test_all_chunks_with_metadata(store):
    store.add("a.txt", CORPUS[:2])
    store.add("b.txt", CORPUS[2:])
    assert store.all_chunks() == [
        Chunk(CORPUS[0], "a.txt", 0),
        Chunk(CORPUS[1], "a.txt", 1),
        Chunk(CORPUS[2], "b.txt", 0),
    ]


def test_metadata_on_search_results(store):
    store.add("a.txt", CORPUS[:2])
    store.add("b.txt", CORPUS[2:])
    assert store.search("vector database embeddings", k=1) == [Chunk(CORPUS[2], "b.txt", 0)]


def test_readding_a_source_replaces_it(store):
    store.add("a.txt", CORPUS)
    store.add("a.txt", CORPUS[:1])
    assert store.all_chunks() == [Chunk(CORPUS[0], "a.txt", 0)]


def test_chroma_persists_across_instances(tmp_path):
    path = str(tmp_path / "chroma")
    ChromaVectorStore(path, FakeEmbedder(), "fake:").add("a.txt", CORPUS)
    reopened = ChromaVectorStore(path, FakeEmbedder(), "fake:")
    assert [chunk.text for chunk in reopened.all_chunks()] == CORPUS


def test_reindex_guard_rejects_a_different_embedder(tmp_path):
    path = str(tmp_path / "chroma")
    ChromaVectorStore(path, FakeEmbedder(), "fake:model-a").add("a.txt", CORPUS)
    with pytest.raises(IndexMismatchError, match="re-ingest"):
        ChromaVectorStore(path, FakeEmbedder(), "fake:model-b")


def test_container_builds_chroma_with_fake_embedder(tmp_path):
    settings = Settings(vector_store="chroma", persist_dir=str(tmp_path / "chroma"))
    assert isinstance(_make_store(settings), ChromaVectorStore)


def test_unknown_vector_store_and_embedder_list_known_names():
    with pytest.raises(NotImplementedError, match="chroma"):
        _make_store(Settings(vector_store="nope"))
    with pytest.raises(NotImplementedError, match="gemini"):
        _make_embedder(Settings(embedding_provider="nope"))


def test_fake_embedder_is_deterministic_and_normalised():
    first = FakeEmbedder().embed(["hello world", "xyz"])
    assert first == FakeEmbedder().embed(["hello world", "xyz"])
    assert sum(value * value for value in first[0]) == pytest.approx(1.0)


class StubEmbedModels:
    def __init__(self, failures=0):
        self.batches = []
        self.failures = failures

    def embed_content(self, model, contents):
        if self.failures:
            self.failures -= 1
            raise RateLimit("slow down")
        self.batches.append(list(contents))
        embeddings = [type("E", (), {"values": [float(len(text)), 1.0]})() for text in contents]
        return type("Response", (), {"embeddings": embeddings})()


class RateLimit(Exception):
    code = 429


class StubClient:
    def __init__(self, failures=0):
        self.models = StubEmbedModels(failures)


def test_gemini_embedder_splits_into_batches():
    client = StubClient()
    embedder = GeminiEmbedder(Settings(embedding_model="m"), client=client, batch_size=3)
    texts = [f"text {i}" for i in range(7)]
    vectors = embedder.embed(texts)
    assert [len(batch) for batch in client.models.batches] == [3, 3, 1]
    assert len(vectors) == 7
    assert vectors[0] == [float(len("text 0")), 1.0]


def test_gemini_embedder_retries_rate_limits(monkeypatch):
    monkeypatch.setattr("doc_agent.services.errors.time.sleep", lambda seconds: None)
    client = StubClient(failures=2)
    embedder = GeminiEmbedder(Settings(embedding_model="m", llm_max_retries=3), client=client)
    assert len(embedder.embed(["a", "b"])) == 2


def test_gemini_embedder_wraps_errors(monkeypatch):
    monkeypatch.setattr("doc_agent.services.errors.time.sleep", lambda seconds: None)
    embedder = GeminiEmbedder(
        Settings(embedding_model="m", llm_max_retries=1), client=StubClient(failures=5)
    )
    with pytest.raises(LLMError):
        embedder.embed(["a"])


def test_gemini_embedder_requires_a_model():
    with pytest.raises(LLMError, match="DOC_AGENT_EMBEDDING_MODEL"):
        GeminiEmbedder(Settings(), client=StubClient())


@pytest.mark.real
def test_gemini_embeddings_vietnamese_and_english(gemini_settings):
    vectors = GeminiEmbedder(gemini_settings).embed(
        ["Hôm nay trời nắng đẹp.", "The weather is sunny today."]
    )
    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1]) > 0
