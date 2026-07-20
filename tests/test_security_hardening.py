"""Production hardening security tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from adaptive_rag.api.app import create_app
from adaptive_rag.api.dependencies.container import get_container, reset_container
from adaptive_rag.config.settings import get_settings
from adaptive_rag.domain.validation.collection_id import validate_collection_id


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("ADAPTIVE_RAG_FAKE_EMBEDDER", "1")
    monkeypatch.setenv("ADAPTIVE_RAG_FAKE_RERANKER", "1")
    monkeypatch.setenv("ADAPTIVE_RAG_FAKE_LLM", "1")
    monkeypatch.setenv("STORAGE__UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("STORAGE__INDEX_DIR", str(tmp_path / "indices"))
    monkeypatch.setenv("STORAGE__DATA_DIR", str(tmp_path / "data"))
    reset_container()
    get_settings.cache_clear()
    return TestClient(create_app())


def test_collection_id_validation_rejects_traversal() -> None:
    with pytest.raises(Exception):
        validate_collection_id("../escape")


def test_collection_stats_rejects_invalid_collection_id(client: TestClient) -> None:
    response = client.get("/api/v1/collections/bad%20id/stats")
    assert response.status_code == 422
    assert response.json()["code"] == "invalid_collection_id"


def test_ingest_rejects_path_outside_upload_dir(client: TestClient, tmp_path: Path) -> None:
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"%PDF-1.4 outside")

    response = client.post(
        "/api/v1/ingest",
        json={"source_path": str(outside), "collection_id": "default"},
    )
    assert response.status_code == 403
    assert response.json()["code"] == "path_access_denied"


def test_upload_rejects_non_pdf_content(client: TestClient) -> None:
    response = client.post(
        "/api/v1/ingest/upload",
        data={"collection_id": "default"},
        files={"file": ("bad.pdf", b"NOTPDF", "application/pdf")},
    )
    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


def test_upload_rejects_invalid_mime_type(client: TestClient, sample_pdf: Path) -> None:
    with sample_pdf.open("rb") as handle:
        response = client.post(
            "/api/v1/ingest/upload",
            data={"collection_id": "default"},
            files={"file": ("hr_policy.pdf", handle, "text/plain")},
        )
    assert response.status_code == 422


def test_ready_endpoint_reports_dependency_checks(client: TestClient) -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    payload = response.json()
    assert "ready" in payload
    assert payload["checks"]["embedder_initialized"] is True
    assert payload["checks"]["index_dir_writable"] is True


def test_ingest_accepts_file_inside_upload_dir(
    client: TestClient,
    sample_pdf: Path,
    tmp_path: Path,
) -> None:
    upload_dir = Path(get_container().settings.storage.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / "hr_policy.pdf"
    target.write_bytes(sample_pdf.read_bytes())

    response = client.post(
        "/api/v1/ingest",
        json={"source_path": "hr_policy.pdf", "collection_id": "secured-ingest"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "completed"
