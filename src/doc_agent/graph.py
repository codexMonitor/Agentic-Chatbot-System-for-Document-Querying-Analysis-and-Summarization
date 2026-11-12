import sqlite3
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from doc_agent.config import Settings
from doc_agent.nodes._utils import timed
from doc_agent.nodes.analysis import make_analysis
from doc_agent.nodes.chat import make_chat
from doc_agent.nodes.qa import make_qa
from doc_agent.nodes.rewrite import make_rewrite
from doc_agent.nodes.router import make_router
from doc_agent.nodes.summarize import make_summarize
from doc_agent.services.base import Chunk
from doc_agent.services.container import Services
from doc_agent.state import AgentState


class _DefaultCheckpointer:
    pass


_DEFAULT_CHECKPOINTER = _DefaultCheckpointer()


def route_by_intent(state: AgentState) -> str:
    return state["intent"]


def _make_serde() -> JsonPlusSerializer | None:
    # Newer LangGraph versions only deserialize explicitly allowed types from checkpoints.
    try:
        return JsonPlusSerializer(allowed_msgpack_modules=[(Chunk.__module__, Chunk.__name__)])
    except TypeError:
        return None


def memory_checkpointer() -> MemorySaver:
    """In-memory checkpointer (tests, throwaway sessions) with the shared serializer config."""
    return MemorySaver(serde=_make_serde())


def sqlite_checkpointer(path: Path) -> SqliteSaver:
    """File-backed checkpointer with the shared serializer config; creates parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False is safe here: SqliteSaver serialises access with its own lock.
    connection = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(connection, serde=_make_serde())


def build_graph(
    services: Services,
    settings: Settings | None = None,
    checkpointer: BaseCheckpointSaver | None | _DefaultCheckpointer = _DEFAULT_CHECKPOINTER,
):
    settings = settings or Settings()
    if checkpointer is _DEFAULT_CHECKPOINTER:
        checkpointer = sqlite_checkpointer(settings.checkpoint_path)
    graph = StateGraph(AgentState)

    def add_node(name: str, run) -> None:
        graph.add_node(name, timed(name, run))

    add_node("rewrite", make_rewrite(services.llm, history_turns=settings.rewrite_history_turns))
    add_node("router", make_router(services.llm))
    add_node("chat", make_chat(services.llm))
    add_node("qa", make_qa(services.llm, services.store, k=settings.retrieval_k))
    add_node("analysis", make_analysis(services.llm, services.store, k=settings.retrieval_k * 2))
    add_node(
        "summarize",
        make_summarize(
            services.llm,
            services.store,
            workers=settings.summarize_workers,
            group_size=settings.summarize_group_size,
        ),
    )
    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "router")
    graph.add_conditional_edges(
        "router",
        route_by_intent,
        {"chat": "chat", "qa": "qa", "analysis": "analysis", "summarize": "summarize"},
    )
    for name in ("chat", "qa", "analysis", "summarize"):
        graph.add_edge(name, END)
    return graph.compile(checkpointer=checkpointer)
