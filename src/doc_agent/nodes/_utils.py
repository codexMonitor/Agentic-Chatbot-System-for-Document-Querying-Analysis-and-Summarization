import logging
import re
import time
from collections.abc import Callable

from langchain_core.messages import AIMessage

from doc_agent.services.base import Chunk
from doc_agent.services.errors import LLMError, RequestDeadlineExceeded
from doc_agent.state import AgentState

logger = logging.getLogger(__name__)

LLM_UNAVAILABLE = "The language model is unavailable. Please try again."
SUMMARIZE_TIMEOUT = (
    "The document may be too long to summarize within the time limit. Please try again later."
)
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def guard_llm_errors(name: str, fn: Callable[[AgentState], dict]) -> Callable[[AgentState], dict]:
    """Turn an `LLMError` that survived the retries into a friendly reply instead of a crash."""

    def run(state: AgentState) -> dict:
        try:
            return fn(state)
        except LLMError as error:
            # Only the (key-masked) error text is logged, never the prompt.
            logger.warning("node=%s failed: %s", name, error)
            message = (
                SUMMARIZE_TIMEOUT
                if name == "summarize"
                and isinstance(error, RequestDeadlineExceeded)
                and error.intent == "summarize"
                else LLM_UNAVAILABLE
            )
            return reply(message, context=[], sources=[])

    return run


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


def question_text(state: AgentState) -> str:
    """The question to act on: the rewritten standalone one if present, else the last message."""
    return state.get("standalone_question") or last_user_text(state)


def check_citations(answer: str, n_sources: int) -> list[int]:
    """Return each cited source number that falls outside the available 1-based source range."""
    if n_sources < 0:
        raise ValueError("n_sources must not be negative")
    return [
        number
        for match in _CITATION_PATTERN.finditer(answer)
        if not 1 <= (number := int(match.group(1))) <= n_sources
    ]


def warn_invalid_citations(answer: str, n_sources: int, node: str) -> None:
    invalid = check_citations(answer, n_sources)
    if invalid:
        logger.warning(
            "node=%s returned invalid citations=%s (source count=%d)", node, invalid, n_sources
        )


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
