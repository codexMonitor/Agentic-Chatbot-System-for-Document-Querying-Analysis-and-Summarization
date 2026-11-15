import json
import sys
from pathlib import Path
from types import ModuleType

import httpx
import pytest

from doc_agent.app_service import DocAgentApp
from doc_agent.config import Settings
from doc_agent.ui import app as ui_app
from doc_agent.ui.app import HTTPClient, InProcessClient, UIClientError, create_client

_APP_FILE = Path(__file__).parents[1] / "src" / "doc_agent" / "ui" / "app.py"


class FakeApp:
    def __init__(self):
        self.calls = []

    def upload_document(self, filename, content):
        self.calls.append(("upload", filename, content))
        return {"source": filename, "chunks": 1}

    def list_documents(self):
        return ["Báo cáo 2026.txt"]

    def delete_document(self, source):
        self.calls.append(("delete-document", source))
        return 1

    def chat(self, thread_id, message):
        self.calls.append(("chat", thread_id, message))
        return {"answer": "Answer", "sources": []}

    def list_threads(self):
        return []

    def get_thread(self, thread_id):
        return {"thread_id": thread_id, "messages": []}

    def delete_thread(self, thread_id):
        return True


def test_in_process_client_uses_the_app_service_and_utf8_filename():
    app = FakeApp()
    client = InProcessClient(app)
    filename = "Báo cáo tiến độ.txt"

    assert client.upload_document(filename, "Nội dung".encode()) == {
        "source": filename,
        "chunks": 1,
    }
    assert app.calls == [("upload", filename, "Nội dung".encode())]
    assert client.list_documents() == ["Báo cáo 2026.txt"]
    assert client.delete_document(filename) == 1
    assert client.chat("thread-1", "Question") == {"answer": "Answer", "sources": []}
    assert client.list_threads() == []
    assert client.get_thread("thread-1") == {"thread_id": "thread-1", "messages": []}
    assert client.delete_thread("thread-1")


def test_http_client_maps_chat_and_thread_operations():
    def respond(request):
        if request.method == "POST":
            return httpx.Response(200, json={"answer": "Answer", "sources": []})
        if request.method == "GET" and request.url.path == "/threads":
            return httpx.Response(
                200,
                json={
                    "threads": [{"thread_id": "thread-1", "title": "Question", "updated_at": None}]
                },
            )
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "thread_id": "thread-1",
                    "title": "Question",
                    "updated_at": None,
                    "messages": [{"role": "user", "content": "Question"}],
                },
            )
        return httpx.Response(200, json={"thread_id": "thread-1", "deleted": True})

    client = HTTPClient("http://localhost:8000", 660, transport=httpx.MockTransport(respond))

    assert client.chat("thread-1", "Question") == {"answer": "Answer", "sources": []}
    assert client.list_threads() == [
        {"thread_id": "thread-1", "title": "Question", "updated_at": None}
    ]
    assert client.get_thread("thread-1")["messages"] == [{"role": "user", "content": "Question"}]
    assert client.delete_thread("thread-1")


def test_http_client_lists_providers_and_sends_model_choice():
    def respond(request):
        if request.url.path == "/providers":
            return httpx.Response(
                200, json={"providers": [{"provider": "gemini", "model": "gemini-test"}]}
            )
        assert json.loads(request.content) == {
            "thread_id": "thread",
            "message": "Hello",
            "provider": "anthropic",
            "model": "claude-test",
        }
        return httpx.Response(200, json={"answer": "Hello", "provider": "anthropic"})

    client = HTTPClient("http://localhost:8000", 660, transport=httpx.MockTransport(respond))
    assert client.list_providers() == [{"provider": "gemini", "model": "gemini-test"}]
    assert client.chat("thread", "Hello", "anthropic", "claude-test")["provider"] == "anthropic"


def test_http_client_shows_provider_access_error():
    def respond(request):
        return httpx.Response(
            403, json={"detail": "Authentication failed. Switch to another provider."}
        )

    client = HTTPClient("http://localhost:8000", 660, transport=httpx.MockTransport(respond))
    with pytest.raises(UIClientError, match="Switch to another provider"):
        client.chat("thread", "Hello", "anthropic")


def test_http_client_encodes_document_names_and_maps_timeout():
    seen_requests = []

    def respond(request):
        seen_requests.append(request)
        if request.method == "POST":
            return httpx.Response(200, json={"source": "Báo cáo 2026.txt", "chunks": 1})
        raise httpx.ReadTimeout("timed out", request=request)

    client = HTTPClient(
        "http://localhost:8000",
        660,
        transport=httpx.MockTransport(respond),
    )
    filename = "Báo cáo 2026.txt"
    assert client.upload_document(filename, b"content")["source"] == filename
    assert filename.encode() in seen_requests[0].content

    with pytest.raises(UIClientError, match="timed out"):
        client.list_documents()


def test_create_embedded_client_uses_fake_services(tmp_path):
    settings = Settings(
        ui_mode="embedded",
        checkpoint_db=str(tmp_path / "checkpoints.sqlite"),
    )

    client = create_client(settings)

    assert isinstance(client, InProcessClient)
    assert isinstance(client._app, DocAgentApp)
    assert client.list_documents() == []
    client._app.close()


def test_ui_timeout_must_exceed_summarize_budget():
    settings = Settings(summarize_timeout_s=600, ui_timeout_s=600)

    with pytest.raises(ValueError, match="longer than DOC_AGENT_SUMMARIZE_TIMEOUT_S"):
        create_client(settings)


def test_ui_startup_shows_actionable_message_for_client_error(monkeypatch):
    class StopUI(Exception):
        pass

    errors = []
    streamlit = ModuleType("streamlit")
    streamlit.session_state = {}
    streamlit.set_page_config = lambda **kwargs: None
    streamlit.title = lambda title: None
    streamlit.error = errors.append
    streamlit.stop = lambda: (_ for _ in ()).throw(StopUI())

    def fail_to_create_client():
        raise UIClientError("The API refused the connection.")

    monkeypatch.setitem(sys.modules, "streamlit", streamlit)
    monkeypatch.setattr(ui_app, "create_client", fail_to_create_client)
    with pytest.raises(StopUI):
        ui_app.main()

    assert len(errors) == 1
    assert "Could not connect to the document service" in errors[0]
    assert "The API refused the connection." in errors[0]
    assert "Check that the API is running" in errors[0]
    assert "DOC_AGENT_MODE=embedded" in errors[0]


def test_embedded_ui_starts_with_fake_services(tmp_path, monkeypatch):
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(tmp_path)
    for name in (
        "DOC_AGENT_MODE",
        "DOC_AGENT_API_URL",
        "DOC_AGENT_UI_TIMEOUT_S",
        "DOC_AGENT_LLM_PROVIDER",
        "DOC_AGENT_VECTOR_STORE",
        "DOC_AGENT_PERSIST_DIR",
        "DOC_AGENT_CHECKPOINT_DB",
        "GEMINI_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DOC_AGENT_MODE", "embedded")
    monkeypatch.setenv("DOC_AGENT_LLM_PROVIDER", "fake")
    monkeypatch.setenv("DOC_AGENT_VECTOR_STORE", "memory")
    monkeypatch.setenv("DOC_AGENT_CHECKPOINT_DB", str(tmp_path / "checkpoints.sqlite"))

    app = AppTest.from_file(str(_APP_FILE)).run(timeout=30)

    assert not app.exception
    assert app.title[0].value == "Document assistant"
    assert app.selectbox[0].value == "fake"
    assert not any("configured LLM provider" in item.value for item in app.warning)
