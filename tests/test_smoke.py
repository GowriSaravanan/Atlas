"""Phase 0 smoke tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from adaptive_rag.api.app import create_app
from adaptive_rag.api.dependencies.container import reset_container
from adaptive_rag.application.workflow.ingest_pipeline import compile_ingest_graph
from adaptive_rag.application.workflow.query_graph import compile_query_graph
from adaptive_rag.application.workflow.state import initial_ingest_state, initial_rag_state
from adaptive_rag.config.settings import Settings, get_settings
from adaptive_rag.domain.models.confidence import ConfidenceBreakdown, ConfidenceScore
from adaptive_rag.domain.models.retrieval import SearchScope


@pytest.fixture(autouse=True)
def _reset_container() -> None:
    """Isolate tests by resetting the DI container."""
    reset_container()
    get_settings.cache_clear()


@pytest.fixture
def client() -> TestClient:
    """FastAPI test client."""
    return TestClient(create_app())


def test_settings_load_with_defaults() -> None:
    """Settings should load with sensible defaults."""
    settings = Settings()
    assert settings.app_name == "Adaptive Hybrid RAG Platform"
    assert settings.conversation.max_turns == 5
    assert settings.retrieval.confidence_threshold == 0.65
    assert settings.llm.provider == "ollama"


def test_search_scope_model() -> None:
    """Domain models should be constructible and typed."""
    scope = SearchScope(filters={"department": "HR"}, estimated_corpus_size=42)
    assert scope.filters["department"] == "HR"
    assert scope.estimated_corpus_size == 42


def test_confidence_breakdown_model() -> None:
    """Confidence breakdown should support explainability fields."""
    breakdown = ConfidenceBreakdown(
        reranker_score=0.9,
        reranker_margin=0.2,
        retrieval_overlap=0.5,
        metadata_match=0.8,
        evidence_density=0.7,
    )
    score = ConfidenceScore(
        value=0.82,
        is_acceptable=True,
        threshold=0.65,
        breakdown=breakdown,
        weights={"reranker_score": 0.3},
    )
    assert score.breakdown.reranker_score == 0.9
    assert score.is_acceptable is True


def test_query_graph_compiles() -> None:
    """Query workflow graph should compile without errors."""
    graph = compile_query_graph()
    state = initial_rag_state(raw_query="What is the leave policy?", conversation_id="test")
    result = graph.invoke(state)
    assert result["raw_query"] == "What is the leave policy?"
    assert result["conversation_id"] == "test"


def test_ingest_graph_compiles() -> None:
    """Ingest workflow graph should compile and run."""
    graph = compile_ingest_graph()
    state = initial_ingest_state(source_path="/tmp/sample.pdf")
    result = graph.invoke(state)
    assert result["status"] == "completed"
    assert result["source_path"] == "/tmp/sample.pdf"


def test_health_endpoint(client: TestClient) -> None:
    """Health endpoint should return ok."""
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["app_name"] == "Adaptive Hybrid RAG Platform"


def test_query_endpoint_skeleton(client: TestClient) -> None:
    """Query endpoint should execute skeleton workflow."""
    response = client.post(
        "/api/v1/query",
        json={"query": "Summarize the HR handbook", "conversation_id": "demo"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer_mode"] == "full"
    assert "trace" in payload
