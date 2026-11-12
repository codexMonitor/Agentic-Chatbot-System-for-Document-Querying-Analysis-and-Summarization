from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from doc_agent.config import Settings
from doc_agent.services.base import Embedder, LLMService, VectorStore
from doc_agent.services.embeddings import EMBEDDERS
from doc_agent.services.llm import LLM_PROVIDERS
from doc_agent.services.vectorstore import ChromaVectorStore, InMemoryVectorStore

T = TypeVar("T")


@dataclass
class Services:
    llm: LLMService
    store: VectorStore


def _lookup(registry: dict[str, T], name: str, kind: str) -> T:
    entry = registry.get(name)
    if entry is None:
        known = ", ".join(sorted(registry))
        raise NotImplementedError(f"Unknown {kind} '{name}'. Known {kind}s: {known}.")
    return entry


def _make_llm(settings: Settings) -> LLMService:
    return _lookup(LLM_PROVIDERS, settings.llm_provider, "LLM provider")(settings)


def _make_embedder(settings: Settings) -> Embedder:
    return _lookup(EMBEDDERS, settings.embedding_provider, "embedding provider")(settings)


def _make_chroma(settings: Settings) -> VectorStore:
    embedder_id = f"{settings.embedding_provider}:{settings.embedding_model}"
    return ChromaVectorStore(settings.persist_dir, _make_embedder(settings), embedder_id)


VECTOR_STORES: dict[str, Callable[[Settings], VectorStore]] = {
    "memory": lambda settings: InMemoryVectorStore(),
    "chroma": _make_chroma,
}


def _make_store(settings: Settings) -> VectorStore:
    return _lookup(VECTOR_STORES, settings.vector_store, "vector store")(settings)


def build_services(settings: Settings) -> Services:
    return Services(llm=_make_llm(settings), store=_make_store(settings))
