import pytest
from fastapi.testclient import TestClient

from doc_agent.api.main import create_app
from doc_agent.config import Settings
from doc_agent.services import Services
from doc_agent.services.llm import FakeLLM
from doc_agent.services.vectorstore import InMemoryVectorStore

_DOCUMENT_TEXT = b"Scrum organizes work into sprints."


def _client(tmp_path, *, llm=None, store=None):
    settings = Settings(
        checkpoint_db=str(tmp_path / "checkpoints.sqlite"),
        max_upload_mb=1,
    )
    services = Services(llm=llm or FakeLLM(), store=store or InMemoryVectorStore())
    return TestClient(create_app(services, settings))


def test_health_upload_list_chat_and_delete(tmp_path):
    with _client(tmp_path) as client:
        assert client.get("/health").json() == {"status": "ok"}

        uploaded = client.post(
            "/documents",
            files={"file": ("../notes.txt", _DOCUMENT_TEXT, "text/plain")},
        )
        assert uploaded.status_code == 200
        assert uploaded.json() == {"source": "notes.txt", "chunks": 1}
        assert client.get("/documents").json() == {"documents": ["notes.txt"]}

        response = client.post(
            "/chat", json={"thread_id": "test-thread", "message": "What is Scrum?"}
        )
        assert response.status_code == 200
        assert response.json() == {
            "intent": "qa",
            "answer": "[fake-llm:qa] Context: [1] (notes.txt#0) Scrum organizes work into sprints. "
            "Question: What is Scrum?",
            "sources": [{"n": 1, "source": "notes.txt", "position": 0}],
        }

        deleted = client.delete("/documents/notes.txt")
        assert deleted.status_code == 200
        assert deleted.json() == {"removed": 1}
        assert client.get("/documents").json() == {"documents": []}


def test_upload_rejects_unsupported_type(tmp_path):
    with _client(tmp_path) as client:
        response = client.post(
            "/documents", files={"file": ("payload.exe", b"no", "application/octet-stream")}
        )
    assert response.status_code == 415


def test_upload_rejects_oversized_content(tmp_path):
    with _client(tmp_path) as client:
        response = client.post(
            "/documents",
            files={"file": ("large.txt", b"x" * (1024 * 1024 + 1), "text/plain")},
        )
    assert response.status_code == 413


def test_upload_reports_invalid_utf8(tmp_path):
    with _client(tmp_path) as client:
        response = client.post("/documents", files={"file": ("bad.txt", b"\xff", "text/plain")})
    assert response.status_code == 400


def test_chat_rejects_blank_messages(tmp_path):
    with _client(tmp_path) as client:
        response = client.post("/chat", json={"thread_id": "t1", "message": "  "})
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("question", "task"),
    [
        ("What is Scrum?", "qa"),
        ("Compare Scrum and LangGraph", "analysis"),
        ("Hello", "chat"),
    ],
)
def test_chat_deadline_covers_qa_analysis_and_chat(tmp_path, monkeypatch, question, task):
    now = [0.0]
    monkeypatch.setattr("doc_agent.services.errors._monotonic", lambda: now[0])

    class SlowLLM(FakeLLM):
        def invoke(self, prompt):
            answer = super().invoke(prompt)
            if f"### TASK: {task}" in prompt:
                now[0] += 121
            return answer

    store = InMemoryVectorStore()
    store.add("notes.txt", ["Scrum organizes work into sprints. LangGraph builds workflows."])
    with _client(tmp_path, llm=SlowLLM(), store=store) as client:
        response = client.post("/chat", json={"thread_id": "slow", "message": question})
    assert response.status_code == 200
    assert (
        response.json()["intent"]
        == {
            "qa": "qa",
            "analysis": "analysis",
            "chat": "chat",
        }[task]
    )
    assert response.json()["answer"] == "The language model is unavailable. Please try again."


def test_summarize_deadline_has_a_friendly_document_length_message(tmp_path, monkeypatch):
    now = [0.0]
    monkeypatch.setattr("doc_agent.services.errors._monotonic", lambda: now[0])

    class SlowSummaryLLM(FakeLLM):
        def invoke(self, prompt):
            answer = super().invoke(prompt)
            if "### TASK: summarize_chunk" in prompt:
                now[0] += 601
            return answer

    store = InMemoryVectorStore()
    store.add("notes.txt", ["Scrum organizes work into sprints."])
    with _client(tmp_path, llm=SlowSummaryLLM(), store=store) as client:
        response = client.post(
            "/chat", json={"thread_id": "slow-summary", "message": "Summarize the document"}
        )
    assert response.status_code == 200
    assert response.json()["intent"] == "summarize"
    assert response.json()["answer"] == (
        "The document may be too long to summarize within the time limit. Please try again later."
    )


def test_chat_deadline_is_shared_across_llm_calls(tmp_path, monkeypatch):
    now = [0.0]
    monkeypatch.setattr("doc_agent.services.errors._monotonic", lambda: now[0])

    class SlowLLM(FakeLLM):
        def invoke(self, prompt):
            answer = super().invoke(prompt)
            if "### TASK: rewrite" in prompt:
                now[0] += 119
            elif "### TASK: qa" in prompt:
                now[0] += 2
            return answer

    store = InMemoryVectorStore()
    store.add("notes.txt", ["Scrum organizes work into sprints."])
    with _client(tmp_path, llm=SlowLLM(), store=store) as client:
        client.post("/chat", json={"thread_id": "shared", "message": "What is Scrum?"})
        response = client.post(
            "/chat", json={"thread_id": "shared", "message": "What are Scrum teams?"}
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "The language model is unavailable. Please try again."
