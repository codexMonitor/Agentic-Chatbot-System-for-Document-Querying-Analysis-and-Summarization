from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from doc_agent.config import Settings
from doc_agent.nodes.analysis import make_analysis
from doc_agent.nodes.qa import make_qa
from doc_agent.nodes.router import make_router
from doc_agent.nodes.summarize import make_summarize
from doc_agent.services.base import Chunk
from doc_agent.services.container import Services
from doc_agent.state import AgentState


def route_by_intent(state: AgentState) -> str:
    return state["intent"]


def _make_checkpointer() -> MemorySaver:
    # Newer LangGraph versions only deserialize explicitly allowed types from checkpoints.
    try:
        serde = JsonPlusSerializer(allowed_msgpack_modules=[(Chunk.__module__, Chunk.__name__)])
    except TypeError:
        return MemorySaver()
    return MemorySaver(serde=serde)


def build_graph(services: Services, settings: Settings | None = None):
    settings = settings or Settings()
    graph = StateGraph(AgentState)
    graph.add_node("router", make_router(services.llm))
    graph.add_node("qa", make_qa(services.llm, services.store, k=settings.retrieval_k))
    graph.add_node(
        "analysis", make_analysis(services.llm, services.store, k=settings.retrieval_k * 2)
    )
    graph.add_node(
        "summarize",
        make_summarize(
            services.llm,
            services.store,
            workers=settings.summarize_workers,
            group_size=settings.summarize_group_size,
            max_retries=settings.llm_max_retries,
        ),
    )
    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_by_intent,
        {"qa": "qa", "analysis": "analysis", "summarize": "summarize"},
    )
    for name in ("qa", "analysis", "summarize"):
        graph.add_edge(name, END)
    return graph.compile(checkpointer=_make_checkpointer())
