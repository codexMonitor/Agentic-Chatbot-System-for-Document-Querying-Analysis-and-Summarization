import threading
import time

import pytest

from doc_agent.nodes.summarize import make_summarize
from doc_agent.services.errors import LLMError
from doc_agent.services.vectorstore import InMemoryVectorStore


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
    llm = CountingLLM(fail_once_on="chunk-042")
    result = summarize(llm, 100, workers=4, group_size=8)
    assert llm.map_calls == 101  # chunk-042 was tried twice
    assert result["answer"].split() == [f"Schunk-{i:03d}" for i in range(100)]


def test_non_retryable_error_stops_the_summary_and_cancels_pending_work():
    class Broken:
        calls = 0

        def invoke(self, prompt):
            Broken.calls += 1
            raise LLMError("API key rejected")

    with pytest.raises(LLMError, match="rejected"):
        summarize(Broken(), 50, workers=1)
    assert Broken.calls < 50


@pytest.mark.parametrize(("workers", "group_size"), [(0, 8), (2, 1)])
def test_invalid_settings_are_rejected(workers, group_size):
    with pytest.raises(ValueError):
        make_summarize(CountingLLM(), make_store(1), workers=workers, group_size=group_size)
