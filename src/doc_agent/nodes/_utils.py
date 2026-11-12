import logging
import time
from collections.abc import Callable

from langchain_core.messages import AIMessage

from doc_agent.services.base import Chunk
from doc_agent.state import AgentState

logger = logging.getLogger(__name__)


def timed(name: str, fn: Callable[[AgentState], dict]) -> Callable[[AgentState], dict]:
    """Wrap a node so each run logs `node=<name> duration_ms=<n>`; the return value is unchanged."""

    def run(state: AgentState) -> dict:
        start = time.perf_counter()
        try:
            return fn(state)
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000)
            logger.info("node=%s duration_ms=%d", name, duration_ms)

    return run


def last_user_text(state: AgentState) -> str:
    messages = state.get("messages") or []
    if not messages:
        return ""
    content = messages[-1].content
    return content if isinstance(content, str) else str(content)


def reply(answer: str, **extra) -> dict:
    return {"messages": [AIMessage(content=answer)], "answer": answer, **extra}


def format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(
        f"[{i}] ({chunk.source}#{chunk.position}) {chunk.text}"
        for i, chunk in enumerate(chunks, start=1)
    )


def source_refs(chunks: list[Chunk]) -> list[dict]:
    """Numbering matches `format_context`, so `[n]` in an answer maps to `sources[n - 1]`."""
    return [
        {"n": i, "source": chunk.source, "position": chunk.position}
        for i, chunk in enumerate(chunks, start=1)
    ]
