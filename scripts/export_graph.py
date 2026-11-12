from pathlib import Path

from doc_agent.config import Settings
from doc_agent.graph import build_graph
from doc_agent.services import Services
from doc_agent.services.llm import FakeLLM
from doc_agent.services.vectorstore import InMemoryVectorStore


def main() -> int:
    graph = build_graph(
        Services(llm=FakeLLM(), store=InMemoryVectorStore()), Settings(), checkpointer=None
    )
    output = Path(__file__).resolve().parents[1] / "docs" / "graph.mmd"
    output.write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
