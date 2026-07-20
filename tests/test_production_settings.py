"""Production settings and env alias tests."""

from __future__ import annotations

import pytest

from adaptive_rag.config.settings import get_settings


def test_flat_env_aliases_apply(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_MODEL", "custom/embedder")
    monkeypatch.setenv("RERANKER_MODEL", "custom/reranker")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "vendor/model")
    get_settings.cache_clear()

    settings = get_settings()
    assert settings.embedding.model_name == "custom/embedder"
    assert settings.reranker.model_name == "custom/reranker"
    assert settings.llm.provider == "openrouter"
    assert settings.llm.model == "vendor/model"
    assert settings.llm.openrouter_api_key == "or-key"

    get_settings.cache_clear()


def test_production_defaults_use_bge_and_openrouter(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "EMBEDDING_MODEL",
        "RERANKER_MODEL",
        "OPENROUTER_API_KEY",
        "OPENROUTER_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()

    settings = get_settings()
    assert settings.embedding.model_name == "BAAI/bge-base-en-v1.5"
    assert settings.reranker.model_name == "BAAI/bge-reranker-base"
    assert settings.llm.provider == "openrouter"

    get_settings.cache_clear()
