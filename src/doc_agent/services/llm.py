import re
from collections.abc import Callable
from typing import Any

from langsmith import traceable

from doc_agent.config import Settings
from doc_agent.services._gemini import make_gemini_client
from doc_agent.services.base import LLMService
from doc_agent.services.errors import LLMError, call_with_retry, mask_secrets

_TASK_RE = re.compile(r"### TASK: (\w+)")
_SUMMARIZE_HINTS = ("summar", "tl;dr", "overview")
_ANALYSIS_HINTS = ("analy", "compare", "extract", "key points", "pros and cons")
_CHAT_PATTERNS = (
    r"(?:hello|hi|hey|good (?:morning|afternoon|evening)|xin chào|chào)"
    r"(?: (?:there|everyone|bạn|anh|chị))?",
    r"(?:thanks|thank you|many thanks)"
    r"(?: (?:very much|so much|a lot|for (?:your|the) (?:help|assistance)))?",
    r"cảm ơn(?: (?:bạn|anh|chị|rất|nhiều|lắm))*",
    r"(?:great|good|okay|ok|got it|understood|được rồi|đã hiểu)"
    r"(?: (?:that|this) (?:helps|helped|is helpful|s helpful))?",
    r"(?:that|this) (?:helps|helped|is helpful|s helpful)",
    r"(?:who are you|what can you do|how can you help|bạn có thể làm gì)",
    r"how (?:do|can|should) i (?:use|open|start|upload|add|ask)"
    r"(?: (?:a |an |the |my |this |your )?"
    r"(?:document|file|assistant|chatbot|app|application|system|conversation|question))?"
    r"(?: to (?:the |this )?(?:assistant|chatbot|app|application|system))?",
)


def _is_chat_question(question: str) -> bool:
    normalized = " ".join(re.sub(r"[^\w\s]", " ", question.lower()).split())
    return any(re.fullmatch(pattern, normalized) for pattern in _CHAT_PATTERNS)


def _trace_inputs(inputs: dict) -> dict:
    prompt = str(inputs.get("prompt", ""))
    return {"messages": [{"role": "user", "content": mask_secrets(prompt)}]}


def _usage_metadata(response: Any) -> dict[str, int] | None:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None

    names = {
        "input_tokens": ("input_tokens", "prompt_token_count", "prompt_tokens"),
        "output_tokens": ("output_tokens", "candidates_token_count", "completion_tokens"),
        "total_tokens": ("total_tokens", "total_token_count"),
    }
    counts: dict[str, int] = {}
    for target, candidates in names.items():
        for name in candidates:
            value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
            if isinstance(value, int) and not isinstance(value, bool):
                counts[target] = value
                break
    return counts or None


def _trace_chat_output(response: Any) -> dict:
    text = getattr(response, "text", None) or ""
    output = {"messages": [{"role": "assistant", "content": mask_secrets(text)}]}
    usage = _usage_metadata(response)
    if usage is not None:
        output["usage_metadata"] = usage
    return output


class FakeLLM:
    """Deterministic offline LLM driven by the `### TASK: <name>` header."""

    def invoke(self, prompt: str) -> str:
        header, _, payload = prompt.partition("\n---\n")
        match = _TASK_RE.search(header)
        task = match.group(1) if match else "unknown"
        if task == "route":
            text = payload.lower()
            if _is_chat_question(payload):
                return "chat"
            if any(hint in text for hint in _SUMMARIZE_HINTS):
                return "summarize"
            if any(hint in text for hint in _ANALYSIS_HINTS):
                return "analysis"
            return "qa"
        if task == "chat":
            if not _is_chat_question(payload):
                return "I can only help with your uploaded documents."
            return (
                "I can answer questions about, analyze, and summarize uploaded documents "
                "with citations."
            )
        if task == "rewrite":
            return payload.rpartition("New question:")[2].strip()
        return f"[fake-llm:{task}] " + " ".join(payload.split())[:200]


class GeminiLLM:
    """Google Gemini (AI Studio) chat model. The key is read from GEMINI_API_KEY."""

    def __init__(self, settings: Settings, client: Any = None) -> None:
        if not settings.llm_model:
            raise LLMError("DOC_AGENT_LLM_MODEL is not set: choose a Gemini model name.")
        self._model = settings.llm_model
        self._max_retries = settings.llm_max_retries
        self._client = client if client is not None else make_gemini_client(settings)
        self._traced_generate_content = traceable(
            self._generate_content,
            run_type="llm",
            name="Gemini chat",
            metadata={"ls_provider": "google_genai", "ls_model_name": self._model},
            process_inputs=_trace_inputs,
            process_outputs=_trace_chat_output,
            exceptions_to_handle=(LLMError,),
        )

    def invoke(self, prompt: str) -> str:
        response = self._traced_generate_content(prompt)
        text = getattr(response, "text", None)
        if not text:
            raise LLMError(f"Gemini model '{self._model}' returned an empty response.")
        return text

    def _generate_content(self, prompt: str) -> Any:
        return call_with_retry(
            lambda: self._client.models.generate_content(model=self._model, contents=prompt),
            max_retries=self._max_retries,
            what=f"Gemini model '{self._model}'",
        )


LLM_PROVIDERS: dict[str, Callable[[Settings], LLMService]] = {
    "fake": lambda settings: FakeLLM(),
    "gemini": GeminiLLM,
}
