from doc_agent.config import load_settings
from doc_agent.graph import build_graph
from doc_agent.services import build_services

settings = load_settings()
graph = build_graph(build_services(settings), settings, checkpointer=None)
