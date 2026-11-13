import re
from pathlib import Path
from statistics import mean
from typing import Any

from doc_agent.nodes._utils import check_citations

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def calculate_metrics(
    results: list[dict[str, Any]], node_latencies_ms: dict[str, list[float]]
) -> dict[str, Any]:
    if not results:
        return {
            "intent_accuracy": None,
            "retrieval_hit_rate": None,
            "invalid_citation_rate": None,
            "avg_latency_ms": {"per_question": None, "per_node": {}},
        }

    intent_accuracy = sum(
        result["intent"] == result["expected_intent"] for result in results
    ) / len(results)
    retrieval_hits = 0
    citation_count = 0
    invalid_count = 0
    for result in results:
        expected = {Path(source).name.casefold() for source in result["source_docs"]}
        actual = {source["source"].casefold() for source in result["retrieved_sources"]}
        retrieval_hits += bool(expected & actual)
        answer = result["answer"]
        citation_count += len(_CITATION_PATTERN.findall(answer))
        invalid_count += len(check_citations(answer, len(result["sources"])))

    per_node = {
        node: round(mean(durations), 2)
        for node, durations in sorted(node_latencies_ms.items())
        if durations
    }
    return {
        "intent_accuracy": intent_accuracy,
        "retrieval_hit_rate": retrieval_hits / len(results),
        "invalid_citation_rate": invalid_count / citation_count if citation_count else 0.0,
        "avg_latency_ms": {
            "per_question": round(mean(result["latency_ms"] for result in results), 2),
            "per_node": per_node,
        },
    }
