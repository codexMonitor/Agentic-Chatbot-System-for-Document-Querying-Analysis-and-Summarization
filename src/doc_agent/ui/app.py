from __future__ import annotations

import hashlib
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any, Protocol
from urllib.parse import quote, urlsplit

import httpx

from doc_agent.app_service import DocAgentApp
from doc_agent.config import Settings, load_settings
from doc_agent.services import build_services
from doc_agent.services.errors import LLMError


class UIClientError(Exception):
    """A user-facing error from the document service."""


class Client(Protocol):
    def upload_document(self, filename: str, content: bytes) -> dict[str, Any]: ...

    def list_documents(self) -> list[str]: ...

    def delete_document(self, source: str) -> int: ...

    def chat(self, thread_id: str, message: str) -> dict[str, Any]: ...

    def list_threads(self) -> list[dict[str, Any]]: ...

    def get_thread(self, thread_id: str) -> dict[str, Any]: ...

    def delete_thread(self, thread_id: str) -> bool: ...


class HTTPClient:
    def __init__(
        self,
        api_url: str,
        timeout_s: float,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        parsed_url = urlsplit(api_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("DOC_AGENT_API_URL must be an absolute HTTP or HTTPS URL")
        if timeout_s <= 0:
            raise ValueError("DOC_AGENT_UI_TIMEOUT_S must be positive")
        self._client = httpx.Client(
            base_url=api_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_s),
            transport=transport,
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as error:
            raise UIClientError(
                "The request timed out while waiting for the document service. "
                "Long summaries can take several minutes; please retry."
            ) from error
        except httpx.ConnectError as error:
            raise UIClientError(
                "The document service is unavailable. Check that the API is running "
                "and that DOC_AGENT_API_URL is correct."
            ) from error
        except httpx.RequestError as error:
            raise UIClientError(
                "Could not reach the document service. Please try again."
            ) from error

        if not response.is_success:
            detail = ""
            try:
                body = response.json()
            except ValueError:
                body = {}
            if isinstance(body, dict) and isinstance(body.get("detail"), str):
                detail = body["detail"]
            if response.status_code >= 500:
                message = "The document service is unavailable. Please try again."
            elif response.status_code == 404:
                message = detail or "That document or conversation was not found."
            else:
                message = detail or (
                    f"The document service rejected the request ({response.status_code})."
                )
            raise UIClientError(message)

        try:
            result = response.json()
        except ValueError as error:
            raise UIClientError("The document service returned an invalid response.") from error
        if not isinstance(result, dict):
            raise UIClientError("The document service returned an invalid response.")
        return result

    def upload_document(self, filename: str, content: bytes) -> dict[str, Any]:
        return self._request(
            "POST",
            "/documents",
            files={"file": (filename, content, "application/octet-stream")},
        )

    def list_documents(self) -> list[str]:
        return self._request("GET", "/documents")["documents"]

    def delete_document(self, source: str) -> int:
        result = self._request("DELETE", f"/documents/{quote(source, safe='')}")
        return result["removed"]

    def chat(self, thread_id: str, message: str) -> dict[str, Any]:
        return self._request("POST", "/chat", json={"thread_id": thread_id, "message": message})

    def list_threads(self) -> list[dict[str, Any]]:
        return self._request("GET", "/threads")["threads"]

    def get_thread(self, thread_id: str) -> dict[str, Any]:
        return self._request("GET", f"/threads/{quote(thread_id, safe='')}")

    def delete_thread(self, thread_id: str) -> bool:
        return self._request("DELETE", f"/threads/{quote(thread_id, safe='')}")["deleted"]


class InProcessClient:
    def __init__(self, app: DocAgentApp) -> None:
        self._app = app

    def _call(self, operation: Callable[..., Any], *args: Any) -> Any:
        try:
            return operation(*args)
        except ValueError as error:
            raise UIClientError(str(error)) from error

    def upload_document(self, filename: str, content: bytes) -> dict[str, Any]:
        return self._call(self._app.upload_document, filename, content)

    def list_documents(self) -> list[str]:
        return self._app.list_documents()

    def delete_document(self, source: str) -> int:
        return self._app.delete_document(source)

    def chat(self, thread_id: str, message: str) -> dict[str, Any]:
        return self._call(self._app.chat, thread_id, message)

    def list_threads(self) -> list[dict[str, Any]]:
        return self._app.list_threads()

    def get_thread(self, thread_id: str) -> dict[str, Any]:
        thread = self._app.get_thread(thread_id)
        if thread is None:
            raise UIClientError("That conversation was not found.")
        return thread

    def delete_thread(self, thread_id: str) -> bool:
        return self._app.delete_thread(thread_id)


def create_client(settings: Settings | None = None) -> Client:
    active_settings = settings or load_settings()
    if active_settings.ui_timeout_s <= active_settings.summarize_timeout_s:
        raise ValueError("DOC_AGENT_UI_TIMEOUT_S must be longer than DOC_AGENT_SUMMARIZE_TIMEOUT_S")
    if active_settings.ui_mode == "embedded":
        return InProcessClient(DocAgentApp(build_services(active_settings), active_settings))
    if active_settings.ui_mode == "http":
        return HTTPClient(active_settings.api_url, active_settings.ui_timeout_s)
    raise ValueError("DOC_AGENT_MODE must be 'http' or 'embedded'")


def _show_message(st: Any, message: dict[str, Any]) -> None:
    role = message["role"]
    with st.chat_message(role):
        if message.get("is_error"):
            st.warning(message["content"])
        else:
            st.markdown(message["content"])
        sources = message.get("sources") or []
        if sources:
            with st.expander("Sources"):
                for source in sources:
                    st.markdown(
                        f"[{source['n']}] **{source['source']}** (chunk {source['position']})"
                    )


def _format_thread_date(value: str | None) -> str:
    if not value:
        return "Date unavailable"
    try:
        return (
            datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone().strftime("%Y-%m-%d")
        )
    except ValueError:
        return value[:10]


def main(client: Client | None = None) -> None:
    import streamlit as st

    st.set_page_config(page_title="Document assistant", page_icon="📚")
    st.title("Document assistant")
    st.warning(
        "Documents and questions are sent to the configured LLM provider. "
        "Please use public documents only."
    )

    if client is None and "_doc_agent_client" not in st.session_state:
        try:
            st.session_state["_doc_agent_client"] = create_client()
        except (LLMError, UIClientError, ValueError) as error:
            st.error(f"Could not start the document assistant: {error}")
            st.stop()
    if client is None:
        client = st.session_state["_doc_agent_client"]

    if "active_thread_id" not in st.session_state:
        st.session_state["active_thread_id"] = str(uuid.uuid4())
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    with st.sidebar:
        st.header("Conversations")
        if st.button("New conversation", use_container_width=True):
            st.session_state["active_thread_id"] = str(uuid.uuid4())
            st.session_state["chat_messages"] = []
            st.rerun()

        try:
            threads = client.list_threads()
        except UIClientError as error:
            st.error(str(error))
            threads = []
        for thread in threads:
            thread_id = thread["thread_id"]
            title = thread.get("title") or "Untitled conversation"
            label = f"{title[:48]} · {_format_thread_date(thread.get('updated_at'))}"
            open_column, delete_column = st.columns([5, 1])
            if open_column.button(
                label,
                key=f"open-thread-{thread_id}",
                use_container_width=True,
                help=title,
            ):
                try:
                    opened = client.get_thread(thread_id)
                except UIClientError as error:
                    st.error(str(error))
                else:
                    st.session_state["active_thread_id"] = thread_id
                    st.session_state["chat_messages"] = [
                        {
                            "role": message["role"],
                            "content": message["content"],
                            "sources": message.get("sources", []),
                        }
                        for message in opened["messages"]
                    ]
                    st.rerun()
            if delete_column.button(
                "Delete",
                key=f"delete-thread-{thread_id}",
                help="Delete this conversation",
            ):
                try:
                    client.delete_thread(thread_id)
                except UIClientError as error:
                    st.error(str(error))
                else:
                    if st.session_state["active_thread_id"] == thread_id:
                        st.session_state["active_thread_id"] = str(uuid.uuid4())
                        st.session_state["chat_messages"] = []
                    st.rerun()

        st.divider()
        st.header("Documents")
        uploaded = st.file_uploader(
            "Upload documents",
            type=["txt", "md", "pdf", "docx"],
            accept_multiple_files=True,
        )
        if st.button("Upload selected files", use_container_width=True):
            for file in uploaded or []:
                content = file.getvalue()
                try:
                    client.upload_document(file.name, content)
                except UIClientError as error:
                    st.error(f"Could not upload {file.name}: {error}")
                else:
                    st.success(f"Uploaded {file.name}")

        try:
            documents = client.list_documents()
        except UIClientError as error:
            st.error(str(error))
            documents = []
        for source in documents:
            source_key = hashlib.sha256(source.encode("utf-8")).hexdigest()
            source_column, remove_column = st.columns([5, 1])
            source_column.caption(source)
            if remove_column.button(
                "Delete",
                key=f"delete-document-{source_key}",
                help=f"Delete {source}",
            ):
                try:
                    client.delete_document(source)
                except UIClientError as error:
                    st.error(str(error))
                else:
                    st.rerun()

    for message in st.session_state["chat_messages"]:
        _show_message(st, message)

    prompt = st.chat_input("Ask a question about your documents")
    if prompt:
        user_message = {"role": "user", "content": prompt}
        st.session_state["chat_messages"].append(user_message)
        _show_message(st, user_message)
        with st.spinner("Thinking... This may take several minutes for a summary."):
            try:
                result = client.chat(st.session_state["active_thread_id"], prompt)
            except UIClientError as error:
                st.error(str(error))
            else:
                answer = result["answer"]
                assistant_message = {
                    "role": "assistant",
                    "content": answer,
                    "sources": result.get("sources", []),
                    "is_error": answer.startswith(
                        (
                            "The language model is unavailable.",
                            "The document may be too long to summarize",
                        )
                    ),
                }
                st.session_state["chat_messages"].append(assistant_message)
                _show_message(st, assistant_message)


if __name__ == "__main__":
    main()
