import json
import logging
from collections import Counter
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from doc_agent.nodes.router import make_router, parse_intent
from doc_agent.prompts import ROUTER
from doc_agent.services.llm import FakeLLM, GeminiLLM

INTENTS_FILE = Path(__file__).parent / "data" / "intents.jsonl"
INTENTS = ("chat", "qa", "analysis", "summarize")


def load_examples() -> list[dict]:
    lines = INTENTS_FILE.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


class FixedLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def invoke(self, prompt: str) -> str:
        return self.reply


def route(llm, question: str) -> str:
    return make_router(llm)({"messages": [HumanMessage(content=question)]})["intent"]


def accuracy(llm, examples: list[dict]) -> float:
    hits = sum(route(llm, item["question"]) == item["intent"] for item in examples)
    return hits / len(examples)


@pytest.mark.parametrize(
    ("raw", "intent"),
    [
        ("qa", "qa"),
        ("analysis", "analysis"),
        ("summarize", "summarize"),
        ("  Summarize.\n", "summarize"),
        ("**analysis**", "analysis"),
        ("Intent: analysis", "analysis"),
        ("summarize\nbecause the user wants an overview", "summarize"),
    ],
)
def test_parse_intent_accepts_valid_output(raw, intent):
    assert parse_intent(raw) == intent


@pytest.mark.parametrize("raw", ["", "   ", "banana", "I am not sure.", "translate"])
def test_parse_intent_falls_back_to_qa(raw):
    assert parse_intent(raw) == "qa"


def test_router_falls_back_to_qa_on_invalid_output():
    assert route(FixedLLM("¯\\_(ツ)_/¯"), "anything") == "qa"


def test_router_logs_chosen_intent(caplog):
    with caplog.at_level(logging.INFO, logger="doc_agent.nodes.router"):
        route(FakeLLM(), "Summarize the document")
    assert "intent=summarize" in caplog.text


def test_router_prompt_has_two_examples_per_intent():
    header = ROUTER.partition("\n---\n")[0]
    counts = Counter(
        line.split(":", 1)[1].strip() for line in header.splitlines() if line.startswith("Intent:")
    )
    assert counts == {intent: 2 for intent in INTENTS}


def test_router_prompt_keeps_task_header_and_separator():
    assert ROUTER.startswith("### TASK: route\n")
    assert "\n---\n{question}" in ROUTER


def test_intents_file_is_well_formed_and_balanced():
    examples = load_examples()
    assert len(examples) == 26
    assert all(set(item) == {"question", "intent"} for item in examples)
    counts = Counter(item["intent"] for item in examples)
    assert set(counts) == set(INTENTS)
    assert min(counts.values()) >= 6


def test_fake_llm_router_accuracy_on_intents_file():
    examples = load_examples()
    score = accuracy(FakeLLM(), examples)
    print(f"router accuracy (FakeLLM, offline): {score:.2f} on {len(examples)} examples")
    # The fake router only knows English keywords, so this is a regression floor, not a quality bar.
    assert score >= 0.75


@pytest.mark.parametrize(
    "question",
    [
        "Hello!",
        "Thanks for your help.",
        "What can you do?",
        "How do I use this assistant?",
        "Xin chào!",
        "Bạn có thể làm gì?",
    ],
)
def test_fake_llm_routes_chat_examples(question):
    assert route(FakeLLM(), question) == "chat"


@pytest.mark.real
def test_real_router_accuracy_on_intents_file(gemini_settings):
    examples = load_examples()
    score = accuracy(GeminiLLM(gemini_settings), examples)
    print(f"router accuracy (real {gemini_settings.llm_model}): {score:.2f} on {len(examples)}")
    assert score >= 0.85
