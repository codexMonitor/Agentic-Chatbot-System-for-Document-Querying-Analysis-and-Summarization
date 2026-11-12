import argparse
import logging
import sys
from pathlib import Path

from doc_agent.config import load_settings
from doc_agent.graph import build_graph
from doc_agent.services import build_services
from doc_agent.services.errors import IndexMismatchError, LLMError
from doc_agent.services.ingestion import ingest_file


def _use_utf8_output() -> None:
    # Windows consoles/pipes default to a legacy code page that cannot print Vietnamese.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _enable_logging() -> logging.Handler:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    logger = logging.getLogger("doc_agent")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return handler


def main(argv: list[str] | None = None) -> int:
    _use_utf8_output()
    parser = argparse.ArgumentParser(prog="doc-agent", description="Chat with your documents.")
    parser.add_argument("files", nargs="*", type=Path, help="documents to ingest")
    parser.add_argument(
        "--thread",
        default="cli",
        help="conversation id; reuse it to continue an earlier conversation (default: cli)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="log routing decisions and node timings"
    )
    args = parser.parse_args(argv)

    handler = _enable_logging() if args.verbose else None
    try:
        return _run(args)
    finally:
        if handler is not None:
            logging.getLogger("doc_agent").removeHandler(handler)


def _run(args: argparse.Namespace) -> int:
    settings = load_settings()
    try:
        services = build_services(settings)
    except (LLMError, IndexMismatchError) as error:
        print(f"! {error}")
        return 1
    for path in args.files:
        try:
            count = ingest_file(path, services.store, settings.chunk_size, settings.chunk_overlap)
        except (OSError, ValueError, LLMError) as error:
            print(f"! skipped {path}: {error}")
            continue
        print(f"ingested {path.name}: {count} chunks")

    graph = build_graph(services, settings)
    config = {"configurable": {"thread_id": args.thread}}
    try:
        _chat_loop(graph, config)
    finally:
        connection = getattr(graph.checkpointer, "conn", None)
        if connection is not None:
            connection.close()
    return 0


def _chat_loop(graph, config: dict) -> None:
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
