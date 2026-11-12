import threading
import time

import pytest

from doc_agent.config import Settings
from doc_agent.nodes.summarize import make_summarize
from doc_agent.services.errors import LLMError, call_with_retry
from doc_agent.services.llm import GeminiLLM
from doc_agent.services.vectorstore import InMemoryVectorStore


class RetryingLLM:
    """Stands in for a provider adapter: it wraps each call in the same `call_with_retry`."""

    def __init__(self, inner, max_retries: int = 3) -> None:
        self.inner = inner
        self.max_retries = max_retries

    def invoke(self, prompt: str) -> str:
        return call_with_retry(
            lambda: self.inner.invoke(prompt), max_retries=self.max_retries, what="test provider"
        )


class CountingLLM:
    """Map calls return `S<text>`; reduce calls join their bullets in the order received."""

    def __init__(self, fail_once_on: str | None = None, jitter: bool = False) -> None:
        self.map_calls = 0
        self.reduce_calls = 0
        self._lock = threading.Lock()
        self._fail_once_on = fail_once_on
        self._jitter = jitter

    def invoke(self, prompt: str) -> str:
        header, _, payload = prompt.partition("\n---\n")
        if "### TASK: summarize_chunk" in header:
            with self._lock:
                self.map_calls += 1
                if self._fail_once_on == payload:
                    self._fail_once_on = None
                    raise LLMError("rate limited", retryable=True)
            if self._jitter:
                # Earlier chunks finish last, so unordered collection would be caught.
                time.sleep(0.02 if payload.endswith(("0", "1")) else 0)
            return f"S{payload}"
        assert "### TASK: summarize_reduce" in header
        with self._lock:
            self.reduce_calls += 1
        return " ".join(line.removeprefix("- ") for line in payload.splitlines())


def make_store(count: int) -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    if count:
        store.add("doc.txt", [f"chunk-{i:03d}" for i in range(count)])
    return store


def summarize(llm, count, **options):
    store = make_store(count)
    return make_summarize(llm, store, **options)({"messages": []})


def test_hundred_chunks_are_all_summarized_without_truncation():
    llm = CountingLLM()
    result = summarize(llm, 100, workers=4, group_size=8)
    assert llm.map_calls == 100
    # 100 -> 13 groups -> 2 groups -> 1: 13 + 2 + 1 merge calls.
    assert llm.reduce_calls == 16
    assert isinstance(result["answer"], str)
    assert result["answer"].split() == [f"Schunk-{i:03d}" for i in range(100)]
    assert len(result["context"]) == 100
    assert result["sources"] == []


def test_order_is_preserved_when_workers_finish_out_of_order():
    result = summarize(CountingLLM(jitter=True), 30, workers=8, group_size=4)
    assert result["answer"].split() == [f"Schunk-{i:03d}" for i in range(30)]


def test_single_chunk_needs_no_merge():
    llm = CountingLLM()
    result = summarize(llm, 1)
    assert result["answer"] == "Schunk-000"
    assert (llm.map_calls, llm.reduce_calls) == (1, 0)


def test_empty_store_keeps_message_and_makes_no_calls():
    llm = CountingLLM()
    result = summarize(llm, 0)
    assert "No documents" in result["answer"]
    assert result["sources"] == [] and result["context"] == []
    assert (llm.map_calls, llm.reduce_calls) == (0, 0)


def test_leftover_single_summary_is_carried_up_without_an_llm_call():
    llm = CountingLLM()
    result = summarize(llm, 9, group_size=8)
    assert llm.reduce_calls == 2  # one group of 8, then the group's result plus the leftover
    assert result["answer"].split() == [f"Schunk-{i:03d}" for i in range(9)]


def test_retryable_failure_in_one_worker_does_not_abort_the_summary(monkeypatch):
    monkeypatch.setattr("doc_agent.services.errors.time.sleep", lambda seconds: None)
    inner = CountingLLM(fail_once_on="chunk-042")
    result = summarize(RetryingLLM(inner), 100, workers=4, group_size=8)
    assert inner.map_calls == 101  # chunk-042 was tried twice
    assert result["answer"].split() == [f"Schunk-{i:03d}" for i in range(100)]


def test_the_node_adds_no_retry_layer_of_its_own():
    inner = CountingLLM(fail_once_on="chunk-003")
    result = summarize(inner, 10, workers=1)
    assert (
        "unavailable" in result["answer"]
    )  # a node-level retry would have recovered from the one failure


def test_non_retryable_error_becomes_a_friendly_answer_and_cancels_pending_work():
    class Broken:
        calls = 0

        def invoke(self, prompt):
            Broken.calls += 1
            raise LLMError("API key rejected")

    result = summarize(Broken(), 50, workers=1)
    assert "unavailable" in result["answer"]
    assert result["sources"] == []
    assert Broken.calls < 50


@pytest.mark.parametrize(("workers", "group_size"), [(0, 8), (2, 1)])
def test_invalid_settings_are_rejected(workers, group_size):
    with pytest.raises(ValueError):
        make_summarize(CountingLLM(), make_store(1), workers=workers, group_size=group_size)


class VirtualClock:
    """A lock-protected logical clock; sleeps advance time without wall-clock scheduling."""

    def __init__(self) -> None:
        self._now = 0.0
        self._lock = threading.Lock()

    def now(self) -> float:
        with self._lock:
            return self._now

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self._now += seconds


class QuotaError(Exception):
    code = 429

    def __init__(self, wait_s: float) -> None:
        super().__init__(f"429 RESOURCE_EXHAUSTED. Please retry in {wait_s:.2f}s.")
        # The structured delay is rounded down, as the real API does.
        self.details = {"error": {"details": [{"retryDelay": f"{int(wait_s)}s"}]}}


class RateLimitedClient:
    """Fake Gemini client: at most `limit` requests per sliding `window_s`, with a retry hint."""

    def __init__(
        self, clock: VirtualClock, inner, limit: int = 15, window_s: float = 60.0, latency_s=2.0
    ):
        self.models = self
        self.rejected = 0
        self.accepted = 0
        self._clock, self._inner = clock, inner
        self._limit, self._window_s, self._latency_s = limit, window_s, latency_s
        self._sent: list[float] = []
        self._lock = threading.Lock()

    def generate_content(self, model, contents):
        with self._lock:
            now = self._clock.now()
            self._sent = [t for t in self._sent if now - t < self._window_s]
            if len(self._sent) >= self._limit:
                self.rejected += 1
                raise QuotaError(self._window_s - (now - self._sent[0]))
            self._sent.append(now)
            self.accepted += 1
        # A real call takes a couple of seconds, which also gives a waiting worker its turn.
        self._clock.sleep(self._latency_s)
        return type("Response", (), {"text": self._inner.invoke(contents)})()


def test_rate_limited_summarization_completes_with_default_settings(monkeypatch):
    clock = VirtualClock()
    monkeypatch.setattr("doc_agent.services.errors.time.sleep", clock.sleep)
    monkeypatch.setattr("doc_agent.services.errors.random.random", lambda: 0.0)
    settings = Settings(llm_model="m")  # default retries, workers and group size
    inner = CountingLLM()
    client = RateLimitedClient(clock, inner)
    llm = GeminiLLM(settings, client=client)
    node = make_summarize(
        llm,
        make_store(45),
        workers=settings.summarize_workers,
        group_size=settings.summarize_group_size,
    )
    result = node({"messages": []})
    assert result["answer"].split() == [f"Schunk-{i:03d}" for i in range(45)]
    assert inner.map_calls == 45
    assert client.rejected > 0  # the limit really was hit and waited out
    assert client.accepted == inner.map_calls + inner.reduce_calls
    assert clock.now() >= 150  # 52 calls at 15 per minute cannot finish sooner
