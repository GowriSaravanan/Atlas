"""Integration tests using the real Sentence Transformers embedder."""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_rag.api.dependencies.container import get_container, reset_container
from adaptive_rag.config.settings import get_settings
from adaptive_rag.domain.models.retrieval import SearchScope


@pytest.fixture
def real_embedder_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Use the real embedder while keeping other fakes for speed."""
    monkeypatch.delenv("ADAPTIVE_RAG_FAKE_EMBEDDER", raising=False)
    monkeypatch.setenv("ADAPTIVE_RAG_FAKE_RERANKER", "1")
    monkeypatch.setenv("ADAPTIVE_RAG_FAKE_LLM", "1")
    monkeypatch.setenv("STORAGE__DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("STORAGE__INDEX_DIR", str(tmp_path / "data" / "indices"))
    monkeypatch.setenv("STORAGE__UPLOAD_DIR", str(tmp_path / "data" / "uploads"))
    reset_container()
    get_settings.cache_clear()
    yield
    reset_container()
    get_settings.cache_clear()


@pytest.mark.integration
def test_real_embedder_returns_semantic_hits(
    real_embedder_env,
    sample_pdf: Path,
) -> None:
    container = get_container()
    container.ingest_document_use_case.execute(
        source_path=str(sample_pdf),
        collection_id="real-embedder",
    )

    scope = SearchScope(filters={"collection_id": "real-embedder"})
    dense_hits = container.index_registry.search_dense(
        "real-embedder",
        "annual leave for employees",
        scope,
        top_k=3,
    )

    assert len(dense_hits) >= 1
    assert any("leave" in hit.chunk.content.lower() for hit in dense_hits)
