import hashlib
import math
import re
from collections.abc import Callable
from typing import Any

from doc_agent.config import Settings
from doc_agent.services._gemini import make_gemini_client
from doc_agent.services.base import Embedder
from doc_agent.services.errors import LLMError, call_with_retry

_TOKEN_RE = re.compile(r"\w+")


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

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            response = call_with_retry(
                lambda batch=batch: self._client.models.embed_content(
                    model=self._model, contents=batch
                ),
                max_retries=self._max_retries,
                what=f"Gemini embedding model '{self._model}'",
            )
            embeddings = getattr(response, "embeddings", None) or []
            if len(embeddings) != len(batch):
                raise LLMError(
                    f"Gemini embedding model '{self._model}' returned {len(embeddings)} "
                    f"vectors for {len(batch)} texts."
                )
            vectors.extend([float(value) for value in item.values] for item in embeddings)
        return vectors


EMBEDDERS: dict[str, Callable[[Settings], Embedder]] = {
    "fake": lambda settings: FakeEmbedder(),
    "gemini": GeminiEmbedder,
}
