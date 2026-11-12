import logging
import random

import pytest

from doc_agent import cli
from doc_agent.config import Settings
from doc_agent.graph import build_graph, memory_checkpointer
from doc_agent.nodes._utils import LLM_UNAVAILABLE
from doc_agent.nodes.summarize import make_summarize
from doc_agent.services import Services
from doc_agent.services.embeddings import GeminiEmbedder
from doc_agent.services.errors import LLMError, call_with_retry, retry_after_from
from doc_agent.services.llm import FakeLLM, GeminiLLM
from doc_agent.services.vectorstore import InMemoryVectorStore

SECRET = "sk-test-0123456789-secret"


class HttpError(Exception):
    def __init__(self, code: int, message: str = "http error") -> None:
        super().__init__(message)
        self.code = code


class StubModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def generate_content(self, model, contents):
        self.calls += 1
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return type("Response", (), {"text": outcome})()

    def embed_content(self, model, contents):
        self.calls += 1
        raise self.outcomes[0]


class StubClient:
    def __init__(self, outcomes):
        self.models = StubModels(outcomes)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("doc_agent.services.errors.time.sleep", lambda seconds: None)


def make_llm(outcomes, max_retries=3):
    client = StubClient(outcomes)
    return GeminiLLM(Settings(llm_model="m", llm_max_retries=max_retries), client=client), client


def make_graph(llm, store=None):
    if store is None:
        store = InMemoryVectorStore()
        store.add("notes.txt", ["Scrum organizes work into sprints."])
    return build_graph(Services(llm=llm, store=store), Settings(), memory_checkpointer())


def ask(graph, question):
    return graph.invoke({"messages": [("user", question)]}, {"configurable": {"thread_id": "t"}})


class DownLLM:
    def __init__(self, error=None):
        self.error = error or LLMError("provider down")
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        raise self.error


@pytest.mark.parametrize("failures", [0, 1, 3])
def test_transient_errors_are_retried_until_success(failures):
    llm, client = make_llm([HttpError(503)] * failures + ["fine"], max_retries=3)
    assert llm.invoke("hi") == "fine"
    assert client.models.calls == failures + 1


def test_max_retries_setting_is_honoured():
    llm, client = make_llm([HttpError(429)], max_retries=1)
    with pytest.raises(LLMError) as info:
        llm.invoke("hi")
    assert client.models.calls == 2
    assert info.value.retryable


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_authentication_and_client_errors_are_not_retried(code):
    llm, client = make_llm([HttpError(code)], max_retries=3)
    with pytest.raises(LLMError) as info:
        llm.invoke("hi")
    assert client.models.calls == 1
    assert not info.value.retryable


def test_embedder_uses_the_same_retry_budget():
    client = StubClient([HttpError(429)])
    embedder = GeminiEmbedder(Settings(embedding_model="e", llm_max_retries=2), client=client)
    with pytest.raises(LLMError):
        embedder.embed(["a"])
    assert client.models.calls == 3


class QuotaError(HttpError):
    """429 shaped like google-genai's `ClientError`: structured details plus a message hint."""

    def __init__(self, retry_delay=None, message_hint=None) -> None:
        text = "429 RESOURCE_EXHAUSTED."
        if message_hint is not None:
            text += f" Please retry in {message_hint}s."
        super().__init__(429, text)
        info = [] if retry_delay is None else [{"@type": "RetryInfo", "retryDelay": retry_delay}]
        self.details = {"error": {"code": 429, "details": info}}


def run_retry(errors, **options):
    """Run `call_with_retry` over a function that raises `errors` in turn; return the sleeps."""
    pending = list(errors)
    sleeps: list[float] = []

    def func():
        if pending:
            raise pending.pop(0)
        return "ok"

    options.setdefault("max_retries", len(errors))
    options.setdefault("jitter", lambda: 0.0)
    assert call_with_retry(func, what="t", sleep=sleeps.append, **options) == "ok"
    return sleeps


def test_a_persistently_failing_call_makes_max_retries_plus_one_attempts():
    llm, client = make_llm([HttpError(503)], max_retries=3)
    with pytest.raises(LLMError):
        llm.invoke("hi")
    assert client.models.calls == 4


def test_summarize_does_not_square_the_attempts():
    store = InMemoryVectorStore()
    store.add("doc.txt", ["only chunk"])
    llm, client = make_llm([HttpError(429)], max_retries=2)
    result = make_summarize(llm, store, workers=1)({"messages": []})
    assert result["answer"] == LLM_UNAVAILABLE
    assert client.models.calls == 3


def test_the_server_retry_delay_is_waited_out():
    assert run_retry([QuotaError(retry_delay="7s")]) == [7.0]


def test_the_precise_message_hint_beats_the_rounded_down_structured_one():
    assert run_retry([QuotaError(retry_delay="5s", message_hint="5.49")]) == [5.49]


def test_retry_after_header_is_read():
    error = HttpError(429)
    error.response = type("Response", (), {"headers": {"retry-after": "12"}})()
    assert run_retry([error]) == [12.0]


def test_server_delay_is_capped_and_never_busy_loops():
    assert run_retry([QuotaError(retry_delay="500s")]) == [60.0]
    assert run_retry([QuotaError(message_hint="0.05")]) == [1.0]


def test_without_a_hint_the_backoff_doubles_and_is_capped():
    assert run_retry([HttpError(503)] * 3) == [1.0, 2.0, 4.0]
    assert run_retry([HttpError(503)] * 8, max_retries=8)[-2:] == [60.0, 60.0]


def test_zero_or_unreadable_hints_fall_back_to_backoff():
    assert run_retry([QuotaError(retry_delay="0s"), QuotaError(retry_delay="soon")]) == [1.0, 2.0]


def test_jitter_stays_within_a_quarter_of_the_backoff():
    sleeps = run_retry([HttpError(503)] * 3, jitter=lambda: 1.0)
    assert sleeps == [1.25, 2.5, 5.0]
    sleeps = run_retry([HttpError(503)] * 3, jitter=random.random)
    assert all(base <= wait <= base * 1.25 for wait, base in zip(sleeps, [1, 2, 4], strict=True))


def test_total_wait_for_one_call_is_bounded():
    sleeps = run_retry([QuotaError(retry_delay="3600s")] * 3)
    assert sum(sleeps) <= 3 * 60.0


def test_the_real_sdk_error_shape_is_understood():
    from google.genai import errors as genai_errors

    # Body of the 429 the free tier returned in the first real summarization run.
    body = {
        "error": {
            "code": 429,
            "message": "You exceeded your current quota. Please retry in 5.490252276s.",
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {"@type": "type.googleapis.com/google.rpc.Help"},
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "5s"},
            ],
        }
    }
    error = genai_errors.ClientError(429, body)
    assert retry_after_from(error) == pytest.approx(5.490252276)
    assert run_retry([error]) == [pytest.approx(5.490252276)]


def test_the_final_error_carries_the_retry_hint():
    llm, _ = make_llm([QuotaError(retry_delay="9s")], max_retries=0)
    with pytest.raises(LLMError) as info:
        llm.invoke("hi")
    assert info.value.retry_after_s == 9.0 and info.value.retryable


def test_hints_on_client_errors_are_ignored():
    llm, client = make_llm([HttpError(401, "bad key. Please retry in 5s.")], max_retries=3)
    with pytest.raises(LLMError) as info:
        llm.invoke("hi")
    assert client.models.calls == 1
    assert info.value.retry_after_s is None
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise LLMError("busy", retryable=True)
        return "ok"

    assert call_with_retry(flaky, max_retries=3, what="t") == "ok"
    calls.clear()

    def permanent():
        calls.append(1)
        raise LLMError("bad key")

    with pytest.raises(LLMError):
        call_with_retry(permanent, max_retries=3, what="t")
    assert len(calls) == 1


def test_permanent_failure_gives_the_friendly_message():
    result = ask(make_graph(DownLLM()), "What is Scrum?")
    assert result["intent"] == "qa"  # the router fell back; the chosen node reports the outage
    assert result["answer"] == LLM_UNAVAILABLE
    assert result["sources"] == [] and result["context"] == []


class FailingOnTask(FakeLLM):
    def __init__(self, task: str) -> None:
        self.task = task

    def invoke(self, prompt):
        if f"### TASK: {self.task}" in prompt:
            raise LLMError("provider down")
        return super().invoke(prompt)


@pytest.mark.parametrize(
    ("question", "intent", "task"),
    [
        ("What is Scrum?", "qa", "qa"),
        ("Compare Scrum and sprints", "analysis", "analysis"),
        ("Summarize the document", "summarize", "summarize_chunk"),
    ],
)
def test_each_node_reports_an_outage_when_the_router_still_works(question, intent, task):
    result = ask(make_graph(FailingOnTask(task)), question)
    assert result["intent"] == intent
    assert result["answer"] == LLM_UNAVAILABLE
    assert result["sources"] == []


def test_embedding_failure_during_retrieval_is_reported_too():
    class BrokenStore(InMemoryVectorStore):
        def search(self, query, k=4):
            raise LLMError("embedding service down")

    result = ask(make_graph(FakeLLM(), BrokenStore()), "What is Scrum?")
    assert result["answer"] == LLM_UNAVAILABLE


def test_rate_limited_provider_is_retried_then_reported():
    llm, client = make_llm([HttpError(429)], max_retries=2)
    result = ask(make_graph(llm), "What is Scrum?")
    assert result["answer"] == LLM_UNAVAILABLE
    assert client.models.calls == 6  # router and qa each try 1 + 2 retries


def test_graph_keeps_working_after_a_failed_turn():
    graph = make_graph(DownLLM())
    ask(graph, "What is Scrum?")
    assert len(ask(graph, "What is Scrum?")["messages"]) == 4


def test_api_key_is_masked_in_error_text_and_logs(monkeypatch, caplog):
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    llm, _ = make_llm([HttpError(401, f"request to ?key={SECRET} rejected")])
    with caplog.at_level(logging.INFO):
        result = ask(make_graph(llm), "What is Scrum?")
    assert result["answer"] == LLM_UNAVAILABLE
    assert "rejected" in caplog.text
    assert SECRET not in caplog.text
    assert SECRET not in str(LLMError(f"oops {SECRET}"))
    assert "***" in str(LLMError(f"oops {SECRET}"))


def test_prompt_text_is_not_logged_on_failure(caplog):
    graph = make_graph(DownLLM())
    with caplog.at_level(logging.INFO):
        ask(graph, "What is Scrum, the very distinctive framework?")
    assert "distinctive" not in caplog.text
    assert "node=qa failed" in caplog.text


def test_cli_reports_an_embedding_failure_during_ingest(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name in ("DOC_AGENT_LLM_PROVIDER", "DOC_AGENT_VECTOR_STORE"):
        monkeypatch.delenv(name, raising=False)

    def fail(*args, **kwargs):
        raise LLMError("embedding service down")

    monkeypatch.setattr(cli, "ingest_file", fail)
    monkeypatch.setattr("builtins.input", lambda prompt="": "quit")
    assert cli.main([str(tmp_path / "doc.txt")]) == 0
    assert "! skipped" in capsys.readouterr().out
