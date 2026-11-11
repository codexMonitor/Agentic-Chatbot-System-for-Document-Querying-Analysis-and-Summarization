import re

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
