"""Phase 1 ingestion tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from adaptive_rag.api.app import create_app
from adaptive_rag.api.dependencies.container import get_container, reset_container
from adaptive_rag.application.workflow.ingest_pipeline import compile_ingest_graph
from adaptive_rag.application.workflow.state import initial_ingest_state
from adaptive_rag.config.settings import get_settings
from adaptive_rag.domain.models.retrieval import SearchScope
from adaptive_rag.domain.policies.adaptive_chunker import AdaptiveChunker
from adaptive_rag.domain.policies.document_metadata_extractor import DocumentMetadataExtractor
from adaptive_rag.infrastructure.pdf.pymupdf_loader import PyMuPDFLoader


@pytest.fixture
def client() -> TestClient:
    reset_container()
    get_settings.cache_clear()
    return TestClient(create_app())


def test_pymupdf_loader_extracts_sections(sample_pdf: Path) -> None:
    loader = PyMuPDFLoader()
    document = loader.load(str(sample_pdf))

    assert "HR Leave Policy" in document.content
    assert document.metadata["page_count"] == 1
    assert len(document.metadata["sections"]) >= 1


def test_adaptive_chunker_respects_sections(sample_pdf: Path) -> None:
    settings = get_settings()
    loader = PyMuPDFLoader()
    document = loader.load(str(sample_pdf))
    document = DocumentMetadataExtractor().apply_to_document(document)

    chunks = AdaptiveChunker(settings.chunking).chunk(document)
    assert len(chunks) >= 1
    assert all(chunk.token_count > 0 for chunk in chunks)


def test_ingest_graph_indexes_pdf(sample_pdf: Path) -> None:
    container = get_container()
    graph = compile_ingest_graph(container.ingest_context)
    state = initial_ingest_state(source_path=str(sample_pdf), collection_id="test-collection")
    result = graph.invoke(state)

    assert result["status"] == "completed"
    assert result["chunk_count"] >= 1

    stats = container.index_registry.stats("test-collection")
    assert stats.chunk_count >= 1
    assert stats.dense_count >= 1
    assert stats.sparse_count >= 1


def test_ingest_upload_endpoint(client: TestClient, sample_pdf: Path) -> None:
    with sample_pdf.open("rb") as handle:
        response = client.post(
            "/api/v1/ingest/upload",
            data={"collection_id": "upload-collection"},
            files={"file": ("hr_policy.pdf", handle, "application/pdf")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["chunk_count"] >= 1

    stats_response = client.get("/api/v1/collections/upload-collection/stats")
    assert stats_response.status_code == 200
    assert stats_response.json()["chunk_count"] >= 1


def test_bm25_and_faiss_search_after_ingest(sample_pdf: Path) -> None:
    container = get_container()
    container.ingest_document_use_case.execute(
        source_path=str(sample_pdf),
        collection_id="search-collection",
    )

    index = container.index_registry
    search_scope = SearchScope(filters={"collection_id": "search-collection"})

    dense_hits = index.search_dense("search-collection", "annual leave HR", search_scope, top_k=3)
    sparse_hits = index.search_sparse("search-collection", "annual leave HR", search_scope, top_k=3)

    assert len(dense_hits) >= 1
    assert len(sparse_hits) >= 1


def test_chunk_metadata_includes_policy_id_from_eval_corpus(tmp_path: Path) -> None:
    from eval.fixtures.corpus import build_eval_corpus_pdf

    pdf_path = build_eval_corpus_pdf(tmp_path / "eval_corpus.pdf")
    container = get_container()
    container.ingest_document_use_case.execute(
        source_path=str(pdf_path),
        collection_id="policy-id-metadata",
    )

    catalog_scope = SearchScope(filters={"collection_id": "policy-id-metadata"})
    hits = container.index_registry.search_sparse(
        "policy-id-metadata",
        "HR-203",
        catalog_scope,
        top_k=10,
    )
    policy_chunks = [hit.chunk for hit in hits if hit.chunk.metadata.get("policy_id") == "HR-203"]
    assert policy_chunks, "Expected at least one chunk with policy_id=HR-203 metadata"

    scoped = SearchScope(filters={"collection_id": "policy-id-metadata", "policy_id": "HR-203"})
    filtered = container.index_registry.search_sparse(
        "policy-id-metadata",
        "annual leave",
        scoped,
        top_k=5,
    )
    assert filtered
    assert all(hit.chunk.metadata.get("policy_id") == "HR-203" for hit in filtered)
