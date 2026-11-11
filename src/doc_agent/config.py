import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "fake"
    vector_store: str = "memory"
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_k: int = 4


def load_settings() -> Settings:
    defaults = Settings()
    env = os.environ
    return Settings(
        llm_provider=env.get("DOC_AGENT_LLM_PROVIDER", defaults.llm_provider),
        vector_store=env.get("DOC_AGENT_VECTOR_STORE", defaults.vector_store),
        chunk_size=int(env.get("DOC_AGENT_CHUNK_SIZE", defaults.chunk_size)),
        chunk_overlap=int(env.get("DOC_AGENT_CHUNK_OVERLAP", defaults.chunk_overlap)),
        retrieval_k=int(env.get("DOC_AGENT_RETRIEVAL_K", defaults.retrieval_k)),
    )
