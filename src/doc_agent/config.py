import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "fake"
    llm_model: str = ""
    embedding_provider: str = "fake"
    embedding_model: str = ""
    vector_store: str = "memory"
    persist_dir: str = ".doc_agent_data"
    # Empty means `<persist_dir>/checkpoints.sqlite`.
    checkpoint_db: str = ""
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_k: int = 4
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 3
    summarize_workers: int = 2
    summarize_group_size: int = 8

    @property
    def checkpoint_path(self) -> Path:
        if self.checkpoint_db:
            return Path(self.checkpoint_db)
        return Path(self.persist_dir) / "checkpoints.sqlite"


def load_settings() -> Settings:
    # Variables already set in the real environment always win over `.env`.
    load_dotenv(Path.cwd() / ".env", override=False, encoding="utf-8")
    defaults = Settings()
    env = os.environ
    return Settings(
        llm_provider=env.get("DOC_AGENT_LLM_PROVIDER", defaults.llm_provider),
        llm_model=env.get("DOC_AGENT_LLM_MODEL", defaults.llm_model),
        embedding_provider=env.get("DOC_AGENT_EMBEDDING_PROVIDER", defaults.embedding_provider),
        embedding_model=env.get("DOC_AGENT_EMBEDDING_MODEL", defaults.embedding_model),
        vector_store=env.get("DOC_AGENT_VECTOR_STORE", defaults.vector_store),
        persist_dir=env.get("DOC_AGENT_PERSIST_DIR", defaults.persist_dir),
        checkpoint_db=env.get("DOC_AGENT_CHECKPOINT_DB", defaults.checkpoint_db),
        chunk_size=int(env.get("DOC_AGENT_CHUNK_SIZE", defaults.chunk_size)),
        chunk_overlap=int(env.get("DOC_AGENT_CHUNK_OVERLAP", defaults.chunk_overlap)),
        retrieval_k=int(env.get("DOC_AGENT_RETRIEVAL_K", defaults.retrieval_k)),
        llm_timeout_s=float(env.get("DOC_AGENT_LLM_TIMEOUT_S", defaults.llm_timeout_s)),
        llm_max_retries=int(env.get("DOC_AGENT_LLM_MAX_RETRIES", defaults.llm_max_retries)),
        summarize_workers=int(env.get("DOC_AGENT_SUMMARIZE_WORKERS", defaults.summarize_workers)),
        summarize_group_size=int(
            env.get("DOC_AGENT_SUMMARIZE_GROUP_SIZE", defaults.summarize_group_size)
        ),
    )
