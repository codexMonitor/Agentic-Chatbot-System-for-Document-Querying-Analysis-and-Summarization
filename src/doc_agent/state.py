from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages

Intent = Literal["qa", "analysis", "summarize"]


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    intent: Intent
    context: list[str]
    answer: str
