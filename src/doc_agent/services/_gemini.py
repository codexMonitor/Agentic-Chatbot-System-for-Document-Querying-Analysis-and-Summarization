import os
from typing import Any

from doc_agent.config import Settings
from doc_agent.services.errors import LLMError

GEMINI_KEY_VAR = "GEMINI_API_KEY"


def make_gemini_client(settings: Settings) -> Any:
    """Build a google-genai client; the key comes from the environment only."""
    api_key = os.environ.get(GEMINI_KEY_VAR, "").strip()
    if not api_key:
        raise LLMError(f"{GEMINI_KEY_VAR} is not set: add it to your .env file or environment.")
    from google import genai
    from google.genai import types

    # google-genai expresses the HTTP timeout in milliseconds.
    options = types.HttpOptions(timeout=int(settings.llm_timeout_s * 1000))
    return genai.Client(api_key=api_key, http_options=options)
