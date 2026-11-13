import re
from pathlib import Path
from statistics import mean
from typing import Any

from doc_agent.nodes._utils import check_citations

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_DECIMAL_COMMA_PATTERN = re.compile(r"(?<=\d),(?=\d)")
_LATEX_WORD_PATTERN = re.compile(r"\b(?:sim|text|mathrm|approx)\b")


def _normalize_key_fact_text(text: str) -> str:
    text = text.casefold().translate(str.maketrans("", "", "$\\{}"))
    text = _LATEX_WORD_PATTERN.sub(" ", text)
    text = text.replace("-", " ").replace("_", " ")
    text = _DECIMAL_COMMA_PATTERN.sub(".", text)
    return " ".join(text.split())


def calculate_key_fact_metrics(results: list[dict[str, Any]]) -> dict[str, float | None]:
    answer_rates = []
    context_rates = []
    for result in results:
        key_facts = result.get("key_facts")
        if not key_facts:
            result.pop("missing_key_facts", None)
            continue
        normalized_answer = _normalize_key_fact_text(result["answer"])
        normalized_context = _normalize_key_fact_text(" ".join(result["retrieved_contexts"]))
        missing_answer = [
            fact for fact in key_facts if _normalize_key_fact_text(fact) not in normalized_answer
        ]
        missing_context = [
            fact for fact in key_facts if _normalize_key_fact_text(fact) not in normalized_context
        ]
        result["missing_key_facts"] = {
            "answer": missing_answer,
            "context": missing_context,
        }
        answer_rates.append((len(key_facts) - len(missing_answer)) / len(key_facts))
        context_rates.append((len(key_facts) - len(missing_context)) / len(key_facts))
    return {
        "key_fact_answer_rate": mean(answer_rates) if answer_rates else None,
        "key_fact_context_rate": mean(context_rates) if context_rates else None,
    }


def calculate_metrics(
    results: list[dict[str, Any]], node_latencies_ms: dict[str, list[float]]
) -> dict[str, Any]:
    if not results:
        return {
            "intent_accuracy": None,
            "retrieval_hit_rate": None,
            "invalid_citation_rate": None,
            "key_fact_answer_rate": None,
            "key_fact_context_rate": None,
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
    metrics = {
        "intent_accuracy": intent_accuracy,
        "retrieval_hit_rate": retrieval_hits / len(results),
        "invalid_citation_rate": invalid_count / citation_count if citation_count else 0.0,
        "avg_latency_ms": {
            "per_question": round(mean(result["latency_ms"] for result in results), 2),
            "per_node": per_node,
        },
    }
    metrics.update(calculate_key_fact_metrics(results))
    return metrics
