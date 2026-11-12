from langchain_core.messages import AIMessage

from doc_agent.services.base import Chunk
from doc_agent.state import AgentState


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
