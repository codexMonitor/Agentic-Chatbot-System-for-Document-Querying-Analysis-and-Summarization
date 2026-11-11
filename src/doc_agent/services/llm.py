import re
from collections.abc import Callable
from typing import Any

from doc_agent.config import Settings
from doc_agent.services._gemini import make_gemini_client
from doc_agent.services.base import LLMService
from doc_agent.services.errors import LLMError, call_with_retry

_TASK_RE = re.compile(r"### TASK: (\w+)")
_SUMMARIZE_HINTS = ("summar", "tl;dr", "overview")
_ANALYSIS_HINTS = ("analy", "compare", "extract", "key points", "pros and cons")


class FakeLLM:
    """Deterministic offline LLM driven by the `### TASK: <name>` header."""

    def invoke(self, prompt: str) -> str:
        header, _, payload = prompt.partition("\n---\n")
        match = _TASK_RE.search(header)
        task = match.group(1) if match else "unknown"
        if task == "route":
            text = payload.lower()
            if any(hint in text for hint in _SUMMARIZE_HINTS):
                return "summarize"
            if any(hint in text for hint in _ANALYSIS_HINTS):
                return "analysis"
            return "qa"
        return f"[fake-llm:{task}] " + " ".join(payload.split())[:200]


class GeminiLLM:
    """Google Gemini (AI Studio) chat model. The key is read from GEMINI_API_KEY."""

    def __init__(self, settings: Settings, client: Any = None) -> None:
        if not settings.llm_model:
            raise LLMError("DOC_AGENT_LLM_MODEL is not set: choose a Gemini model name.")
        self._model = settings.llm_model
        self._max_retries = settings.llm_max_retries
        self._client = client if client is not None else make_gemini_client(settings)

    def invoke(self, prompt: str) -> str:
        response = call_with_retry(
            lambda: self._client.models.generate_content(model=self._model, contents=prompt),
            max_retries=self._max_retries,
            what=f"Gemini model '{self._model}'",
        )
        text = getattr(response, "text", None)
        if not text:
            raise LLMError(f"Gemini model '{self._model}' returned an empty response.")
        return text


LLM_PROVIDERS: dict[str, Callable[[Settings], LLMService]] = {
    "fake": lambda settings: FakeLLM(),
    "gemini": GeminiLLM,
}
