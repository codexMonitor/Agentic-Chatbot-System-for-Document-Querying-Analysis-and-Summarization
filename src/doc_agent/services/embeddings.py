import hashlib
import math
import re
from collections.abc import Callable
from typing import Any

from langsmith import traceable

from doc_agent.config import Settings
from doc_agent.services._gemini import make_gemini_client
from doc_agent.services.base import Embedder
from doc_agent.services.errors import LLMError, call_with_retry, mask_secrets

_TOKEN_RE = re.compile(r"\w+")


def _trace_inputs(inputs: dict) -> dict:
    texts = inputs.get("texts", [])
    return {"texts": [mask_secrets(str(text)) for text in texts]}


def _usage_metadata(response: Any) -> dict[str, int] | None:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None

    names = {
        "input_tokens": ("input_tokens", "prompt_token_count", "prompt_tokens"),
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


def _trace_embedding_output(response: Any) -> dict:
    output = {
        "embeddings": [
            {"embedding": [float(value) for value in item.values]}
            for item in (getattr(response, "embeddings", None) or [])
        ]
    }
    usage = _usage_metadata(response)
    if usage is not None:
        output["usage_metadata"] = usage
    return output


class FakeEmbedder:
    """Deterministic offline embedder: hashed bag-of-words, L2-normalised."""

    def __init__(self, dimensions: int = 64) -> None:
        self._dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "big") % self._dimensions] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            vector[0] = 1.0
            return vector
        return [value / norm for value in vector]


class GeminiEmbedder:
    """Gemini embeddings (multilingual). Texts are sent in batches; the key is GEMINI_API_KEY."""

    def __init__(self, settings: Settings, client: Any = None, batch_size: int = 100) -> None:
        if not settings.embedding_model:
            raise LLMError("DOC_AGENT_EMBEDDING_MODEL is not set: choose a Gemini embedding model.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._model = settings.embedding_model
        self._max_retries = settings.llm_max_retries
        self._batch_size = batch_size
        self._client = client if client is not None else make_gemini_client(settings)
        self._traced_embed_content = traceable(
            self._embed_content,
            run_type="embedding",
            name="Gemini embeddings",
            metadata={"ls_provider": "google_genai", "ls_model_name": self._model},
            process_inputs=_trace_inputs,
            process_outputs=_trace_embedding_output,
            exceptions_to_handle=(LLMError,),
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            response = self._traced_embed_content(batch)
            embeddings = getattr(response, "embeddings", None) or []
            if len(embeddings) != len(batch):
                raise LLMError(
                    f"Gemini embedding model '{self._model}' returned {len(embeddings)} "
                    f"vectors for {len(batch)} texts."
                )
            vectors.extend([float(value) for value in item.values] for item in embeddings)
        return vectors

    def _embed_content(self, texts: list[str]) -> Any:
        return call_with_retry(
            lambda: self._client.models.embed_content(model=self._model, contents=texts),
            max_retries=self._max_retries,
            what=f"Gemini embedding model '{self._model}'",
        )


EMBEDDERS: dict[str, Callable[[Settings], Embedder]] = {
    "fake": lambda settings: FakeEmbedder(),
    "gemini": GeminiEmbedder,
}
