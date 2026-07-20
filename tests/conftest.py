"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest


@pytest.fixture(autouse=True)
def _test_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Configure isolated temp storage and fake embedder for tests."""
    monkeypatch.setenv("ADAPTIVE_RAG_FAKE_EMBEDDER", "1")
    monkeypatch.setenv("ADAPTIVE_RAG_FAKE_RERANKER", "1")
    monkeypatch.setenv("STORAGE__DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("STORAGE__INDEX_DIR", str(tmp_path / "data" / "indices"))
    monkeypatch.setenv("STORAGE__UPLOAD_DIR", str(tmp_path / "data" / "uploads"))

    from adaptive_rag.api.dependencies.container import reset_container
    from adaptive_rag.config.settings import get_settings

    reset_container()
    get_settings.cache_clear()


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Create a sample PDF with headings and body text."""
    pdf_path = tmp_path / "hr_policy.pdf"
    document = fitz.open()
    page = document.new_page()

    page.insert_text((72, 72), "HR Leave Policy", fontsize=18)
    page.insert_text(
        (72, 110),
        "The HR department provides 20 days of annual leave for full-time employees.",
        fontsize=11,
    )
    page.insert_text((72, 150), "Sick Leave", fontsize=16)
    page.insert_text(
        (72, 180),
        "Employees may take up to 10 sick leave days per year with manager approval.",
        fontsize=11,
    )

    document.save(pdf_path)
    document.close()
    return pdf_path
