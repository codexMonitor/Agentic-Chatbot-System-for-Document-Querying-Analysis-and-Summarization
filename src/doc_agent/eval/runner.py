import argparse
import json
import logging
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from doc_agent.config import Settings, load_settings
from doc_agent.eval.metrics import calculate_metrics
from doc_agent.graph import build_graph, memory_checkpointer
from doc_agent.services import Services, build_services
from doc_agent.services.base import LLMService
from doc_agent.services.ingestion import LOADERS, load_text, split_text

_REQUIRED_FIELDS = {
    "id",
    "question",
    "reference_answer",
    "source_docs",
    "expected_intent",
}
_INTENTS = {"qa", "analysis", "summarize"}
_NODE_TIMING_PATTERN = re.compile(r"node=([\w-]+) duration_ms=(\d+)")
_JUDGE_METRICS = {
    "faithfulness": None,
    "answer_relevance": None,
    "context_relevance": None,
}
_NODE_LOGGER_NAME = "doc_agent.nodes._utils"


@dataclass(frozen=True)
class _PreparedDocument:
    path: Path
    chunks: list[str]


class _CountingLLM:
    def __init__(self, llm: LLMService) -> None:
        self._llm = llm
        self.calls = 0

    def invoke(self, prompt: str) -> str:
        self.calls += 1
        return self._llm.invoke(prompt)


class _NodeTimingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.durations_ms: dict[str, list[float]] = {}

    def emit(self, record: logging.LogRecord) -> None:
        match = _NODE_TIMING_PATTERN.search(record.getMessage())
        if match:
            node, duration = match.groups()
            self.durations_ms.setdefault(node, []).append(float(duration))


def _read_dataset(path: Path) -> list[dict[str, Any]]:
    rows = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
        if not isinstance(row, dict) or set(row) != _REQUIRED_FIELDS:
            raise ValueError(f"{path}:{line_number}: expected fields {_REQUIRED_FIELDS}")
        if not isinstance(row["id"], str) or not row["id"]:
            raise ValueError(f"{path}:{line_number}: id must be a non-empty string")
        if row["id"] in seen_ids:
            raise ValueError(f"{path}:{line_number}: duplicate id {row['id']!r}")
        seen_ids.add(row["id"])
        for field in ("question", "reference_answer"):
            if not isinstance(row[field], str) or not row[field]:
                raise ValueError(f"{path}:{line_number}: {field} must be a non-empty string")
        if not isinstance(row["source_docs"], list) or not all(
            isinstance(source, str) and source for source in row["source_docs"]
        ):
            raise ValueError(f"{path}:{line_number}: source_docs must be a list of strings")
        if row["expected_intent"] not in _INTENTS:
            raise ValueError(f"{path}:{line_number}: invalid expected_intent")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: dataset has no rows")
    return rows


def _prepare_documents(
    docs_dir: Path, chunk_size: int, chunk_overlap: int
) -> list[_PreparedDocument]:
    if not docs_dir.is_dir():
        raise ValueError(f"documents directory does not exist: {docs_dir}")
    paths = sorted(
        (path for path in docs_dir.iterdir() if path.is_file() and path.suffix.lower() in LOADERS),
        key=lambda path: path.name.casefold(),
    )
    if not paths:
        raise ValueError(f"no supported documents found in {docs_dir}")
    return [
        _PreparedDocument(path, split_text(load_text(path), chunk_size, chunk_overlap))
        for path in paths
    ]


def _reduce_call_count(n_chunks: int, group_size: int) -> int:
    calls = 0
    while n_chunks > 1:
        n_chunks = (n_chunks + group_size - 1) // group_size
        calls += n_chunks
    return calls


def _expected_llm_calls(rows: list[dict[str, Any]], n_chunks: int, group_size: int) -> int:
    calls = 0
    for row in rows:
        calls += 1  # router
        if row["expected_intent"] == "summarize":
            calls += n_chunks + _reduce_call_count(n_chunks, group_size)
        else:
            calls += 1  # QA or analysis response, assuming a non-empty context
    return calls


def _unavailable_judge(reason: str) -> tuple[dict[str, None], dict[str, Any]]:
    return dict(_JUDGE_METRICS), {"status": "unavailable", "reason": reason}


def _judge_results(
    results: list[dict[str, Any]], settings: Settings, requested: bool
) -> tuple[dict[str, float | None], dict[str, Any]]:
    if not requested:
        return dict(_JUDGE_METRICS), {"status": "disabled", "reason": "--judge was not set"}
    if settings.llm_provider != "gemini":
        return _unavailable_judge("--judge requires DOC_AGENT_LLM_PROVIDER=gemini")
    if not os.environ.get("GEMINI_API_KEY"):
        return _unavailable_judge("GEMINI_API_KEY is not configured")
    if not settings.llm_model or not settings.embedding_model:
        return _unavailable_judge(
            "DOC_AGENT_LLM_MODEL and DOC_AGENT_EMBEDDING_MODEL are required for --judge"
        )
    try:
        from ragas.embeddings import GoogleEmbeddings
        from ragas.llms import llm_factory
        from ragas.metrics.collections import AnswerRelevancy, ContextRelevance, Faithfulness
    except ImportError:
        return _unavailable_judge("Ragas is not installed; install the optional `eval` extra")

    from doc_agent.services._gemini import make_gemini_client

    client = make_gemini_client(settings)
    judge_llm = llm_factory(settings.llm_model, provider="google", client=client)
    embeddings = GoogleEmbeddings(client=client, model=settings.embedding_model)
    scorers = {
        "faithfulness": Faithfulness(llm=judge_llm),
        "answer_relevance": AnswerRelevancy(llm=judge_llm, embeddings=embeddings),
        "context_relevance": ContextRelevance(llm=judge_llm),
    }
    scores: dict[str, list[float]] = {name: [] for name in scorers}
    for result in results:
        inputs = {
            "user_input": result["question"],
            "response": result["answer"],
            "retrieved_contexts": result["retrieved_contexts"],
        }
        for name, scorer in scorers.items():
            scores[name].append(float(scorer.score(**inputs).value))
    return (
        {name: mean(values) if values else None for name, values in scores.items()},
        {"status": "completed", "reason": None},
    )


def run_evaluation(
    dataset_path: Path,
    docs_dir: Path,
    settings: Settings,
    *,
    out_path: Path | None = None,
    include_summarize: bool = False,
    limit: int | None = None,
    judge: bool = False,
    services: Services | None = None,
    print_fn: Callable[[str], None] = print,
) -> dict[str, Any]:
    rows = _read_dataset(dataset_path)
    if limit is not None:
        rows = rows[:limit]
    skipped = [
        row for row in rows if not include_summarize and row["expected_intent"] == "summarize"
    ]
    run_rows = [row for row in rows if include_summarize or row["expected_intent"] != "summarize"]
    documents = _prepare_documents(docs_dir, settings.chunk_size, settings.chunk_overlap)
    n_chunks = sum(len(document.chunks) for document in documents)
    expected_calls = _expected_llm_calls(run_rows, n_chunks, settings.summarize_group_size)
    print_fn(
        f"Expected LLM calls: {expected_calls} "
        "(estimate; excludes embeddings and optional Ragas judge calls)."
    )

    services = services or build_services(settings)
    counting_llm = _CountingLLM(services.llm)
    for document in documents:
        services.store.add(document.path.name, document.chunks)
    graph = build_graph(
        Services(llm=counting_llm, store=services.store), settings, memory_checkpointer()
    )

    timing_handler = _NodeTimingHandler()
    node_logger = logging.getLogger(_NODE_LOGGER_NAME)
    original_level = node_logger.level
    node_logger.setLevel(logging.INFO)
    node_logger.addHandler(timing_handler)
    results: list[dict[str, Any]] = []
    try:
        for row in run_rows:
            start = time.perf_counter()
            result = graph.invoke(
                {"messages": [("user", row["question"])]},
                {"configurable": {"thread_id": f"eval-{row['id']}"}},
            )
            latency_ms = (time.perf_counter() - start) * 1000
            chunks = result.get("context") or []
            sources = result.get("sources") or []
            results.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "reference_answer": row["reference_answer"],
                    "source_docs": row["source_docs"],
                    "expected_intent": row["expected_intent"],
                    "intent": result["intent"],
                    "answer": result["answer"],
                    "retrieved_contexts": [chunk.text for chunk in chunks],
                    "retrieved_sources": [
                        {"source": chunk.source, "position": chunk.position} for chunk in chunks
                    ],
                    "sources": sources,
                    "latency_ms": round(latency_ms, 2),
                }
            )
    finally:
        node_logger.removeHandler(timing_handler)
        node_logger.setLevel(original_level)

    metrics = calculate_metrics(results, timing_handler.durations_ms)
    judge_metrics, judge_status = _judge_results(results, settings, judge)
    metrics.update(judge_metrics)
    destination = out_path or (
        Path("eval") / "results" / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "schema_version": 1,
        "settings": {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "k": settings.retrieval_k,
            "provider": settings.llm_provider,
            "model": settings.llm_model,
            "vector_store": settings.vector_store,
            "docs": str(docs_dir),
        },
        "counts": {
            "dataset_rows": len(rows),
            "run": len(results),
            "skipped": len(skipped),
            "llm_calls_expected": expected_calls,
            "llm_calls_actual": counting_llm.calls,
        },
        "metrics": metrics,
        "judge": judge_status,
        "results": results,
    }
    destination.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print_fn(
        f"Actual LLM calls: {counting_llm.calls} (pipeline calls; embeddings and judge excluded)."
    )
    if all(value is None for value in judge_metrics.values()):
        print_fn(f"Judge metrics: null ({judge_status['reason']}).")
    print_fn(f"Results written to {destination}")
    return output


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate doc_agent on a JSONL question set.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--docs", type=Path, required=True, help="directory of documents to ingest")
    parser.add_argument("--include-summarize", action="store_true")
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument("--k", type=_positive_int)
    parser.add_argument("--chunk-size", type=_positive_int)
    parser.add_argument("--chunk-overlap", type=int)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    settings = load_settings()
    overrides: dict[str, int] = {}
    if args.k is not None:
        overrides["retrieval_k"] = args.k
    if args.chunk_size is not None:
        overrides["chunk_size"] = args.chunk_size
    if args.chunk_overlap is not None:
        overrides["chunk_overlap"] = args.chunk_overlap
    settings = replace(settings, **overrides)
    if settings.chunk_size <= 0 or not 0 <= settings.chunk_overlap < settings.chunk_size:
        parser.error("--chunk-overlap must satisfy 0 <= overlap < chunk-size")
    run_evaluation(
        args.dataset,
        args.docs,
        settings,
        out_path=args.out,
        include_summarize=args.include_summarize,
        limit=args.limit,
        judge=args.judge,
    )
    return 0
