import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TypeVar

from doc_agent.config import Settings
from doc_agent.services.base import Embedder, LLMService, VectorStore
from doc_agent.services.embeddings import EMBEDDERS
from doc_agent.services.llm import LLM_PROVIDERS
from doc_agent.services.vectorstore import (
    ChromaVectorStore,
    InMemoryVectorStore,
    SupabaseVectorStore,
)

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


def available_chat_providers(settings: Settings) -> list[dict[str, str]]:
    providers = []
    names = sorted(LLM_PROVIDERS, key=lambda name: (name != "gemini", name))
    for name in names:
        if name == "fake":
            if settings.llm_provider == "fake":
                providers.append({"provider": name, "model": ""})
        elif os.environ.get(f"{name.upper()}_API_KEY", "").strip():
            model = settings.anthropic_model if name == "anthropic" else settings.llm_model
            providers.append({"provider": name, "model": model})
    return providers


def select_chat_settings(settings: Settings, provider: str | None, model: str | None) -> Settings:
    selected = provider or settings.llm_provider
    if selected not in LLM_PROVIDERS:
        raise ValueError(f"Unknown provider '{selected}'.")
    key_name = None if selected == "fake" else f"{selected.upper()}_API_KEY"
    if key_name and not os.environ.get(key_name, "").strip():
        raise ValueError(f"{selected} provider not configured: {key_name} is missing.")
    if model is not None and not model.strip():
        raise ValueError("model must not be empty")
    changes = {"llm_provider": selected}
    if model is not None:
        changes["anthropic_model" if selected == "anthropic" else "llm_model"] = model
    return replace(settings, **changes)


def _make_embedder(settings: Settings) -> Embedder:
    return _lookup(EMBEDDERS, settings.embedding_provider, "embedding provider")(settings)


def _make_chroma(settings: Settings) -> VectorStore:
    embedder_id = f"{settings.embedding_provider}:{settings.embedding_model}"
    return ChromaVectorStore(settings.persist_dir, _make_embedder(settings), embedder_id)


def _make_supabase(settings: Settings) -> VectorStore:
    embedder_id = f"{settings.embedding_provider}:{settings.embedding_model}"
    return SupabaseVectorStore(_make_embedder(settings), embedder_id)


VECTOR_STORES: dict[str, Callable[[Settings], VectorStore]] = {
    "memory": lambda settings: InMemoryVectorStore(),
    "chroma": _make_chroma,
    "supabase": _make_supabase,
}


def _make_store(settings: Settings) -> VectorStore:
    return _lookup(VECTOR_STORES, settings.vector_store, "vector store")(settings)


def build_services(settings: Settings) -> Services:
    return Services(llm=_make_llm(settings), store=_make_store(settings))
