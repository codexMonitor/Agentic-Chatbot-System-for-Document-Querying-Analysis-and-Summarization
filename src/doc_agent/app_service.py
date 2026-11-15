from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_core.messages import AIMessage, HumanMessage

from doc_agent.config import Settings
from doc_agent.graph import build_graph
from doc_agent.services.base import LLMService
from doc_agent.services.container import (
    Services,
    _make_llm,
    available_chat_providers,
    select_chat_settings,
)
from doc_agent.services.errors import (
    ProviderAccessError,
    ensure_request_deadline,
    request_deadline,
)
from doc_agent.services.ingestion import LOADERS, ingest_file, unsupported_type_message


class UnsupportedDocumentTypeError(ValueError):
    pass


class DocumentTooLargeError(ValueError):
    pass


class ProviderRequestError(Exception):
    """A provider needs attention before the user can retry."""


class _DeadlineAwareLLM:
    def __init__(self, llm: LLMService) -> None:
        self._llm = llm

    def invoke(self, prompt: str) -> str:
        ensure_request_deadline()
        try:
            answer = self._llm.invoke(prompt)
        except ProviderAccessError as error:
            raise ProviderRequestError(str(error)) from error
        ensure_request_deadline()
        return answer


class DocAgentApp:
    def __init__(self, services: Services, settings: Settings) -> None:
        if settings.max_upload_mb < 1:
            raise ValueError("max_upload_mb must be at least 1")
        if settings.chat_timeout_s <= 0 or settings.summarize_timeout_s <= 0:
            raise ValueError("chat and summarize timeouts must be positive")
        self._services = services
        self._settings = settings
        self._graph = build_graph(
            Services(llm=_DeadlineAwareLLM(services.llm), store=services.store), settings
        )

    @property
    def max_upload_bytes(self) -> int:
        return self._settings.max_upload_mb * 1024 * 1024

    def upload_document(self, filename: str, content: bytes) -> dict[str, str | int]:
        source = Path(filename.replace("\\", "/")).name
        if not source or source in {".", ".."}:
            raise ValueError("a filename is required")
        suffix = Path(source).suffix.lower()
        if suffix not in LOADERS:
            raise UnsupportedDocumentTypeError(unsupported_type_message(suffix))
        if len(content) > self.max_upload_bytes:
            raise DocumentTooLargeError(
                f"file exceeds the {self._settings.max_upload_mb} MB upload limit"
            )

        with TemporaryDirectory() as directory:
            path = Path(directory) / source
            path.write_bytes(content)
            chunks = ingest_file(
                path,
                self._services.store,
                self._settings.chunk_size,
                self._settings.chunk_overlap,
                source=source,
                max_upload_mb=self._settings.max_upload_mb,
                max_table_rows=self._settings.max_table_rows,
                max_table_cells=self._settings.max_table_cells,
                include_hidden_sheets=self._settings.include_hidden_sheets,
            )
        return {"source": source, "chunks": chunks}

    def list_documents(self) -> list[str]:
        return self._services.store.sources()

    def delete_document(self, source: str) -> int:
        return self._services.store.delete(source)

    def list_providers(self) -> list[dict[str, str]]:
        return available_chat_providers(self._settings)

    def chat(
        self, thread_id: str, message: str, provider: str | None = None, model: str | None = None
    ) -> dict:
        if not thread_id.strip():
            raise ValueError("thread_id must not be empty")
        if not message.strip():
            raise ValueError("message must not be empty")
        selected = select_chat_settings(self._settings, provider, model)
        model_name = (
            selected.anthropic_model if selected.llm_provider == "anthropic" else selected.llm_model
        )
        llm = self._services.llm if selected == self._settings else _make_llm(selected)
        graph = (
            self._graph
            if selected == self._settings
            else build_graph(
                Services(llm=_DeadlineAwareLLM(llm), store=self._services.store),
                selected,
                checkpointer=self._graph.checkpointer,
            )
        )
        config = {
            "configurable": {"thread_id": thread_id},
            "metadata": {"provider": selected.llm_provider, "model": model_name},
        }
        with request_deadline(self._settings.chat_timeout_s, self._settings.summarize_timeout_s):
            result = graph.invoke({"messages": [("user", message)]}, config)
        return {
            "intent": result["intent"],
            "answer": result["answer"],
            "sources": result.get("sources") or [],
            "provider": selected.llm_provider,
            "model": model_name,
        }

    def list_threads(self) -> list[dict[str, str | None]]:
        thread_ids = {
            checkpoint.config["configurable"]["thread_id"]
            for checkpoint in self._graph.checkpointer.list(None)
        }
        threads = [
            thread for thread_id in thread_ids if (thread := self.get_thread(thread_id)) is not None
        ]
        return sorted(threads, key=lambda thread: thread["updated_at"] or "", reverse=True)

    def get_thread(self, thread_id: str) -> dict | None:
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = self._graph.get_state(config)
        messages = snapshot.values.get("messages") or []
        if not messages:
            return None

        title = next(
            (message.content for message in messages if isinstance(message, HumanMessage)), ""
        )
        serialized_messages = [
            {
                "role": "user" if isinstance(message, HumanMessage) else "assistant",
                "content": message.content
                if isinstance(message, (HumanMessage, AIMessage))
                and isinstance(message.content, str)
                else str(message.content),
                "provider": message.additional_kwargs.get("provider"),
                "model": message.additional_kwargs.get("model"),
            }
            for message in messages
            if isinstance(message, (HumanMessage, AIMessage))
        ]
        return {
            "thread_id": thread_id,
            "title": title if isinstance(title, str) else str(title),
            "updated_at": snapshot.created_at,
            "messages": serialized_messages,
        }

    def delete_thread(self, thread_id: str) -> bool:
        if self.get_thread(thread_id) is None:
            return False
        self._graph.checkpointer.delete_thread(thread_id)
        return True

    def close(self) -> None:
        connection = getattr(self._graph.checkpointer, "conn", None)
        if connection is not None:
            connection.close()
