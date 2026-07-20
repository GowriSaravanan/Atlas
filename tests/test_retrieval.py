"""Phase 2 hybrid retrieval tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from adaptive_rag.api.app import create_app
from adaptive_rag.api.dependencies.container import get_container, reset_container
from adaptive_rag.config.settings import get_settings
from adaptive_rag.domain.models.retrieval import RetrievalStrategy
from adaptive_rag.domain.policies.rrf import ReciprocalRankFusion
from adaptive_rag.domain.models.document import Chunk
from adaptive_rag.domain.models.retrieval import ScoredChunk


@pytest.fixture
def client() -> TestClient:
    reset_container()
    get_settings.cache_clear()
    return TestClient(create_app())


@pytest.fixture
def indexed_collection(sample_pdf: Path) -> str:
    container = get_container()
    collection_id = "retrieval-phase2"
    container.ingest_document_use_case.execute(
        source_path=str(sample_pdf),
        collection_id=collection_id,
    )
    return collection_id


def test_rrf_fuses_ranked_lists() -> None:
    chunk_a = Chunk(id="a", document_id="d1", content="alpha")
    chunk_b = Chunk(id="b", document_id="d1", content="beta")
    chunk_c = Chunk(id="c", document_id="d1", content="gamma")

    dense = [
        ScoredChunk(chunk=chunk_a, score=0.9, source="dense", rank=1),
        ScoredChunk(chunk=chunk_b, score=0.8, source="dense", rank=2),
    ]
    sparse = [
        ScoredChunk(chunk=chunk_b, score=5.0, source="bm25", rank=1),
        ScoredChunk(chunk=chunk_c, score=3.0, source="bm25", rank=2),
    ]

    settings = get_settings()
    fusion = ReciprocalRankFusion(settings.retrieval)
    fused = fusion.fuse({"dense": dense, "bm25": sparse}, "test query")

    assert len(fused) == 3
    assert fused[0].chunk.id == "b"
    assert fused[0].source == "rrf"
    assert fused[0].rank == 1


def test_hybrid_retrieval_returns_fused_results(indexed_collection: str) -> None:
    container = get_container()
    result = container.hybrid_retrieval_use_case.execute(
        query="annual leave HR department",
        collection_id=indexed_collection,
        strategy=RetrievalStrategy.HYBRID,
        top_k=5,
    )

    assert result.strategy == RetrievalStrategy.HYBRID
    assert len(result.results) >= 1
    assert len(result.dense_hits) >= 1
    assert len(result.sparse_hits) >= 1
    assert len(result.fused_hits) >= 1
    assert result.confidence is not None
    assert result.analysis is not None
    assert result.decision is not None
    assert "dense_ms" in result.trace.latency_ms
    assert "sparse_ms" in result.trace.latency_ms
    assert "fusion_ms" in result.trace.latency_ms


def test_bm25_only_strategy(indexed_collection: str) -> None:
    container = get_container()
    result = container.hybrid_retrieval_use_case.execute(
        query="sick leave",
        collection_id=indexed_collection,
        strategy=RetrievalStrategy.BM25,
    )

    assert result.strategy == RetrievalStrategy.BM25
    assert len(result.sparse_hits) >= 1
    assert result.dense_hits == []
    assert result.fused_hits == []
    assert result.results == result.sparse_hits


def test_dense_only_strategy(indexed_collection: str) -> None:
    container = get_container()
    result = container.hybrid_retrieval_use_case.execute(
        query="sick leave employees",
        collection_id=indexed_collection,
        strategy=RetrievalStrategy.DENSE,
    )

    assert result.strategy == RetrievalStrategy.DENSE
    assert len(result.dense_hits) >= 1
    assert result.sparse_hits == []
    assert result.fused_hits == []
    assert result.results == result.dense_hits


def test_retrieve_endpoint(client: TestClient, indexed_collection: str) -> None:
    response = client.post(
        "/api/v1/retrieve",
        json={
            "query": "HR leave policy",
            "collection_id": indexed_collection,
            "strategy": "hybrid",
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["strategy"] == "hybrid"
    assert len(payload["results"]) >= 1
    assert len(payload["trace"]["fused_hits"]) >= 1


def test_retrieve_empty_collection() -> None:
    container = get_container()
    result = container.hybrid_retrieval_use_case.execute(
        query="anything",
        collection_id="nonexistent-collection",
        strategy=RetrievalStrategy.HYBRID,
    )

    assert result.results == []
    assert result.dense_hits == []
    assert result.sparse_hits == []
