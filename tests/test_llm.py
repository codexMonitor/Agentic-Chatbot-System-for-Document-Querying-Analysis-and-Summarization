import os

import pytest
from dotenv import dotenv_values

from conftest import pytest_collection_modifyitems
from doc_agent.config import Settings
from doc_agent.services.container import _make_llm
from doc_agent.services.errors import (
    LLMError,
    ProviderAccessError,
    call_with_retry,
    request_deadline,
    set_request_intent,
)
from doc_agent.services.llm import LLM_PROVIDERS, AnthropicLLM, FakeLLM, GeminiLLM


class RateLimitError(Exception):
    code = 429


class StubModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.configs = []

    def generate_content(self, model, contents, config=None):
        self.calls += 1
        self.configs.append(config)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return type("Response", (), {"text": outcome})()


class StubClient:
    def __init__(self, outcomes):
        self.models = StubModels(outcomes)


def make_llm(outcomes, **overrides):
    settings = Settings(llm_model="stub-model", **overrides)
    client = StubClient(outcomes)
    return GeminiLLM(settings, client=client), client


def test_gemini_returns_text():
    llm, client = make_llm(["OK"])
    assert llm.invoke("Say OK") == "OK"
    assert client.models.calls == 1


def test_gemini_request_timeout_is_limited_by_the_shared_deadline(monkeypatch):
    monkeypatch.setattr("doc_agent.services.errors._monotonic", lambda: 10.0)
    llm, client = make_llm(["OK"])
    with request_deadline(chat_timeout_s=12, summarize_timeout_s=600):
        set_request_intent("qa")
        assert llm.invoke("Say OK") == "OK"
    assert client.models.configs[0].http_options.timeout == 12000


def test_provider_exception_becomes_llm_error():
    llm, _ = make_llm([ValueError("boom")])
    with pytest.raises(LLMError, match="boom"):
        llm.invoke("hi")


def test_rate_limit_is_retried_then_wrapped(monkeypatch):
    monkeypatch.setattr("doc_agent.services.errors.time.sleep", lambda seconds: None)
    llm, client = make_llm([RateLimitError("slow down")] * 3, llm_max_retries=2)
    with pytest.raises(LLMError, match="slow down"):
        llm.invoke("hi")
    assert client.models.calls == 3


def test_rate_limit_recovers_after_retry(monkeypatch):
    monkeypatch.setattr("doc_agent.services.errors.time.sleep", lambda seconds: None)
    llm, client = make_llm([RateLimitError("slow down"), "fine"], llm_max_retries=3)
    assert llm.invoke("hi") == "fine"
    assert client.models.calls == 2


def test_empty_response_is_an_error():
    llm, _ = make_llm([""])
    with pytest.raises(LLMError, match="empty"):
        llm.invoke("hi")


def test_backoff_delays_double():
    delays = []

    def always_fail():
        raise RateLimitError("x")

    with pytest.raises(LLMError):
        call_with_retry(
            always_fail, max_retries=3, what="t", sleep=delays.append, jitter=lambda: 0.0
        )
    assert delays == [1.0, 2.0, 4.0]


def test_missing_key_gives_clear_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        GeminiLLM(Settings(llm_model="stub-model"))


def test_missing_model_gives_clear_error():
    with pytest.raises(LLMError, match="DOC_AGENT_LLM_MODEL"):
        GeminiLLM(Settings(), client=StubClient([]))


def test_unknown_provider_lists_known_names():
    with pytest.raises(NotImplementedError) as info:
        _make_llm(Settings(llm_provider="nope"))
    message = str(info.value)
    assert "nope" in message
    for name in LLM_PROVIDERS:
        assert name in message


def test_fake_provider_is_registered():
    assert isinstance(_make_llm(Settings(llm_provider="fake")), FakeLLM)


def test_real_tests_are_skipped_by_default(monkeypatch):
    monkeypatch.delenv("DOC_AGENT_RUN_REAL", raising=False)

    class Item:
        keywords = {"real": True}

        def __init__(self):
            self.markers = []

        def add_marker(self, marker):
            self.markers.append(marker)

    item = Item()
    pytest_collection_modifyitems(None, [item])
    assert [marker.name for marker in item.markers] == ["skip"]


@pytest.mark.real
def test_gemini_smoke(gemini_settings):
    assert GeminiLLM(gemini_settings).invoke("Say OK").strip()


class AnthropicMessages:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        block = type("Block", (), {"type": "text", "text": outcome})()
        return type("Response", (), {"content": [block], "usage": None})()


class AnthropicStub:
    def __init__(self, outcomes):
        self.messages = AnthropicMessages(outcomes)


def test_anthropic_uses_messages_api_and_single_retry_layer(monkeypatch):
    monkeypatch.setattr("doc_agent.services.errors.time.sleep", lambda seconds: None)
    client = AnthropicStub([RateLimitError("slow down"), "OK"])
    llm = AnthropicLLM(Settings(anthropic_model="test-claude"), client=client)
    assert llm.invoke("Say OK") == "OK"
    assert len(client.messages.calls) == 2
    assert client.messages.calls[-1]["model"] == "test-claude"
    assert client.messages.calls[-1]["messages"] == [{"role": "user", "content": "Say OK"}]


@pytest.mark.parametrize(
    ("status", "text", "expected"),
    [(401, "bad key", "authentication"), (400, "credit balance is too low", "credit")],
)
def test_anthropic_access_errors_are_friendly_and_not_retried(status, text, expected):
    class ProviderFailure(Exception):
        status_code = status

    client = AnthropicStub([ProviderFailure(text)])
    llm = AnthropicLLM(Settings(), client=client)
    with pytest.raises(ProviderAccessError, match=expected) as error:
        llm.invoke("hello")
    assert "switch" in str(error.value)
    assert not error.value.retryable
    assert len(client.messages.calls) == 1


def test_anthropic_missing_key_is_optional(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError, match="provider not configured"):
        _make_llm(Settings(llm_provider="anthropic"))


def test_gemini_authentication_error_suggests_switching():
    class AuthenticationFailure(Exception):
        code = 401

    llm, client = make_llm([AuthenticationFailure("invalid key")])
    with pytest.raises(ProviderAccessError, match="switch"):
        llm.invoke("hello")
    assert client.models.calls == 1


@pytest.mark.real
def test_anthropic_smoke(monkeypatch):
    values = {**dotenv_values(".env"), **os.environ}
    if not values.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY is not set")
    monkeypatch.setenv("ANTHROPIC_API_KEY", values["ANTHROPIC_API_KEY"])
    settings = Settings(
        anthropic_model=values.get("DOC_AGENT_ANTHROPIC_MODEL") or "claude-haiku-5-5"
    )
    assert AnthropicLLM(settings).invoke("Say OK").strip()
