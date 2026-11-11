import os

import pytest
from dotenv import dotenv_values

from doc_agent.config import Settings


def pytest_collection_modifyitems(config, items):
    if os.environ.get("DOC_AGENT_RUN_REAL") == "1":
        return
    skip = pytest.mark.skip(reason="real-provider test: set DOC_AGENT_RUN_REAL=1 to run")
    for item in items:
        if "real" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def gemini_settings(monkeypatch):
    """Settings for real Gemini tests, built from the environment and `.env` (key kept in env)."""
    values = {**dotenv_values(".env"), **os.environ}
    key = values.get("GEMINI_API_KEY")
    if not key:
        pytest.skip("GEMINI_API_KEY is not set")
    monkeypatch.setenv("GEMINI_API_KEY", key)
    return Settings(
        llm_model=values.get("DOC_AGENT_LLM_MODEL") or "gemini-3.5-flash-lite",
        embedding_model=values.get("DOC_AGENT_EMBEDDING_MODEL") or "gemini-embedding-001",
    )
