import argparse
from pathlib import Path

from doc_agent.config import load_settings
from doc_agent.graph import build_graph
from doc_agent.services import build_services
from doc_agent.services.errors import IndexMismatchError, LLMError
from doc_agent.services.ingestion import ingest_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="doc-agent", description="Chat with your documents.")
    parser.add_argument("files", nargs="*", type=Path, help="documents to ingest")
    args = parser.parse_args(argv)

    settings = load_settings()
    try:
        services = build_services(settings)
    except (LLMError, IndexMismatchError) as error:
        print(f"! {error}")
        return 1
    for path in args.files:
        try:
            count = ingest_file(path, services.store, settings.chunk_size, settings.chunk_overlap)
        except (OSError, ValueError) as error:
            print(f"! skipped {path}: {error}")
            continue
        print(f"ingested {path.name}: {count} chunks")

    graph = build_graph(services, settings)
    config = {"configurable": {"thread_id": "cli"}}
    while True:
        try:
            question = input("you> ").strip()
        except EOFError:
            break
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue
        result = graph.invoke({"messages": [("user", question)]}, config)
        print(f"[{result['intent']}] {result['answer']}")
        sources = result.get("sources") or []
        if sources:
            print("Sources:")
            for item in sources:
                print(f"[{item['n']}] {item['source']}#{item['position']}")
    return 0
