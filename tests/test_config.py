import pytest

from doc_agent.config import Settings, load_settings

ENV_VARS = [
    "DOC_AGENT_LLM_PROVIDER",
    "DOC_AGENT_LLM_MODEL",
    "DOC_AGENT_EMBEDDING_PROVIDER",
    "DOC_AGENT_EMBEDDING_MODEL",
    "DOC_AGENT_VECTOR_STORE",
    "DOC_AGENT_PERSIST_DIR",
    "DOC_AGENT_CHUNK_SIZE",
    "DOC_AGENT_CHUNK_OVERLAP",
    "DOC_AGENT_RETRIEVAL_K",
    "DOC_AGENT_LLM_TIMEOUT_S",
    "DOC_AGENT_LLM_MAX_RETRIES",
    "DOC_AGENT_CHECKPOINT_DB",
    "DOC_AGENT_SUMMARIZE_WORKERS",
    "DOC_AGENT_SUMMARIZE_GROUP_SIZE",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for name in ENV_VARS:
        # setenv first so teardown removes anything load_dotenv() adds during the test
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)


def test_defaults():
    settings = load_settings()
    assert settings == Settings()
    assert settings.persist_dir == ".doc_agent_data"
    assert settings.llm_timeout_s == 60.0
    assert settings.llm_max_retries == 3
    assert settings.summarize_workers == 2
    assert settings.summarize_group_size == 8


def test_env_overrides_defaults(monkeypatch):
    monkeypatch.setenv("DOC_AGENT_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("DOC_AGENT_LLM_MODEL", "some-model")
    monkeypatch.setenv("DOC_AGENT_EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("DOC_AGENT_EMBEDDING_MODEL", "some-embedding")
    monkeypatch.setenv("DOC_AGENT_PERSIST_DIR", "store")
    monkeypatch.setenv("DOC_AGENT_LLM_TIMEOUT_S", "12.5")
    monkeypatch.setenv("DOC_AGENT_LLM_MAX_RETRIES", "5")
    monkeypatch.setenv("DOC_AGENT_SUMMARIZE_WORKERS", "6")
    monkeypatch.setenv("DOC_AGENT_SUMMARIZE_GROUP_SIZE", "4")
    settings = load_settings()
    assert settings.llm_provider == "gemini"
    assert settings.llm_model == "some-model"
    assert settings.embedding_provider == "gemini"
    assert settings.embedding_model == "some-embedding"
    assert settings.persist_dir == "store"
    assert settings.llm_timeout_s == 12.5
    assert settings.llm_max_retries == 5
    assert settings.summarize_workers == 6
    assert settings.summarize_group_size == 4


def test_dotenv_file_is_picked_up(tmp_path):
    (tmp_path / ".env").write_text("DOC_AGENT_LLM_MODEL=from-dotenv\n", encoding="utf-8")
    assert load_settings().llm_model == "from-dotenv"


def test_real_environment_wins_over_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("DOC_AGENT_LLM_MODEL=from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("DOC_AGENT_LLM_MODEL", "from-env")
    assert load_settings().llm_model == "from-env"


def test_repr_contains_no_secrets(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-value")
    assert "super-secret-value" not in repr(load_settings())
    assert "key" not in repr(Settings()).lower()
