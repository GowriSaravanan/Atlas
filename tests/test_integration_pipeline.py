"""Integration tests using production AI adapters."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from adaptive_rag.api.dependencies.container import get_container, reset_container
from adaptive_rag.config.settings import get_settings
from adaptive_rag.domain.models.retrieval import SearchScope
from adaptive_rag.infrastructure.embeddings.sentence_transformer import SentenceTransformerEmbedder
from adaptive_rag.infrastructure.llm.openrouter_llm import OpenRouterProviderLLM
from adaptive_rag.infrastructure.reranking.cross_encoder import CrossEncoderReranker


@pytest.fixture
def real_ai_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Disable fake adapters and use configured production models."""
    for env_name in (
        "ADAPTIVE_RAG_FAKE_EMBEDDER",
        "ADAPTIVE_RAG_FAKE_RERANKER",
        "ADAPTIVE_RAG_FAKE_LLM",
    ):
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setenv("STORAGE__DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("STORAGE__INDEX_DIR", str(tmp_path / "data" / "indices"))
    monkeypatch.setenv("STORAGE__UPLOAD_DIR", str(tmp_path / "data" / "uploads"))
    reset_container()
    get_settings.cache_clear()
    yield
    reset_container()
    get_settings.cache_clear()


@pytest.mark.integration
def test_production_embedder_is_sentence_transformer(real_ai_env) -> None:
    container = get_container()
    assert isinstance(container.embedder, SentenceTransformerEmbedder)
    assert container.embedder.model_name == get_settings().embedding.model_name
    assert container.embedder.dimension > 0


@pytest.mark.integration
def test_production_reranker_is_cross_encoder(real_ai_env) -> None:
    container = get_container()
    settings = get_settings()
    assert isinstance(container.reranker, CrossEncoderReranker)
    assert settings.reranker.model_name.startswith("BAAI/bge-reranker")


@pytest.mark.integration
def test_production_llm_is_openrouter_when_configured(real_ai_env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")
    reset_container()
    get_settings.cache_clear()

    container = get_container()
    assert isinstance(container.llm, OpenRouterProviderLLM)
    assert container.llm.model_name == "meta-llama/llama-3.1-8b-instruct"


@pytest.mark.integration
def test_real_embedder_returns_semantic_hits(
    real_ai_env,
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


@pytest.mark.integration
def test_real_reranker_reorders_candidates(real_ai_env) -> None:
    from adaptive_rag.domain.models.document import Chunk
    from adaptive_rag.domain.models.retrieval import ScoredChunk

    container = get_container()
    reranker = container.reranker
    candidates = [
        ScoredChunk(
            chunk=Chunk(id="c1", document_id="d1", content="annual leave policy 20 days"),
            score=0.2,
            source="dense",
            rank=1,
        ),
        ScoredChunk(
            chunk=Chunk(id="c2", document_id="d1", content="sick leave 10 days manager approval"),
            score=0.9,
            source="dense",
            rank=2,
        ),
    ]
    reranked = reranker.rerank("sick leave days", candidates, top_k=2)
    assert [hit.chunk.id for hit in reranked]
    assert reranked[0].source == "reranker"


@pytest.mark.integration
def test_full_pipeline_with_openrouter(
    real_ai_env,
    sample_pdf: Path,
) -> None:
    settings = get_settings()
    if not settings.llm.openrouter_api_key:
        pytest.skip("OPENROUTER_API_KEY required for live OpenRouter integration test")

    container = get_container()
    container.ingest_document_use_case.execute(
        source_path=str(sample_pdf),
        collection_id="production",
    )

    result = container.hybrid_retrieval_use_case.execute(
        query="How many sick leave days are allowed?",
        collection_id="production",
        top_k=5,
    )

    assert isinstance(container.embedder, SentenceTransformerEmbedder)
    assert isinstance(container.reranker, CrossEncoderReranker)
    assert isinstance(container.llm, OpenRouterProviderLLM)
    assert result.generated_answer is not None
    assert result.generated_answer.answer
    assert result.generated_answer.citations
    assert any(step.step == "citation_formatting" for step in result.trace.steps)
