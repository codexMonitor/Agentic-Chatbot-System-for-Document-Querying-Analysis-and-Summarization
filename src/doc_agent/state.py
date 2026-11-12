from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages

from doc_agent.services.base import Chunk

Intent = Literal["qa", "analysis", "summarize"]


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    intent: Intent
    context: list[Chunk]
    sources: list[dict]
    standalone_question: str
    answer: str
