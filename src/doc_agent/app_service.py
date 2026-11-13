from pathlib import Path
from tempfile import TemporaryDirectory

from doc_agent.config import Settings
from doc_agent.graph import build_graph
from doc_agent.services.base import LLMService
from doc_agent.services.container import Services
from doc_agent.services.errors import (
    ensure_request_deadline,
    request_deadline,
)
from doc_agent.services.ingestion import LOADERS, ingest_file


class UnsupportedDocumentTypeError(ValueError):
    pass


class DocumentTooLargeError(ValueError):
    pass


class _DeadlineAwareLLM:
    def __init__(self, llm: LLMService) -> None:
        self._llm = llm

    def invoke(self, prompt: str) -> str:
        ensure_request_deadline()
        answer = self._llm.invoke(prompt)
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
            supported = ", ".join(sorted(LOADERS))
            raise UnsupportedDocumentTypeError(f"unsupported file type (supported: {supported})")
        if len(content) > self.max_upload_bytes:
            raise DocumentTooLargeError(
                f"file exceeds the {self._settings.max_upload_mb} MB upload limit"
            )

        with TemporaryDirectory() as directory:
            path = Path(directory) / f"upload{suffix}"
            path.write_bytes(content)
            chunks = ingest_file(
                path,
                self._services.store,
                self._settings.chunk_size,
                self._settings.chunk_overlap,
                source=source,
            )
        return {"source": source, "chunks": chunks}

    def list_documents(self) -> list[str]:
        return self._services.store.sources()

    def delete_document(self, source: str) -> int:
        return self._services.store.delete(source)

    def chat(self, thread_id: str, message: str) -> dict:
        if not thread_id.strip():
            raise ValueError("thread_id must not be empty")
        if not message.strip():
            raise ValueError("message must not be empty")
        config = {"configurable": {"thread_id": thread_id}}
        with request_deadline(self._settings.chat_timeout_s, self._settings.summarize_timeout_s):
            result = self._graph.invoke({"messages": [("user", message)]}, config)
        return {
            "intent": result["intent"],
            "answer": result["answer"],
            "sources": result.get("sources") or [],
        }

    def close(self) -> None:
        connection = getattr(self._graph.checkpointer, "conn", None)
        if connection is not None:
            connection.close()
