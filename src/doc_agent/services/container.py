from dataclasses import dataclass

from doc_agent.config import Settings
from doc_agent.services.base import LLMService, VectorStore
from doc_agent.services.llm import LLM_PROVIDERS
from doc_agent.services.vectorstore import InMemoryVectorStore


@dataclass
class Services:
    llm: LLMService
    store: VectorStore


def _make_llm(settings: Settings) -> LLMService:
    factory = LLM_PROVIDERS.get(settings.llm_provider)
    if factory is None:
        known = ", ".join(sorted(LLM_PROVIDERS))
        raise NotImplementedError(
            f"Unknown LLM provider '{settings.llm_provider}'. Known providers: {known}."
        )
    return factory(settings)


def _make_store(settings: Settings) -> VectorStore:
    if settings.vector_store == "memory":
        return InMemoryVectorStore()
    raise NotImplementedError(
        f"Unknown vector store '{settings.vector_store}': add an adapter and register it here."
    )


def build_services(settings: Settings) -> Services:
    return Services(llm=_make_llm(settings), store=_make_store(settings))
