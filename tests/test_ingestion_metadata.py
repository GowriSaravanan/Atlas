"""Ingestion metadata extraction tests."""

from __future__ import annotations

from pathlib import Path

from eval.fixtures.catalog import build_chunk_catalog
from eval.fixtures.corpus import build_eval_corpus_pdf

from adaptive_rag.api.dependencies.container import get_container


def test_ingest_extracts_policy_id_chunk_metadata(tmp_path: Path) -> None:
    pdf_path = tmp_path / "eval_corpus.pdf"
    build_eval_corpus_pdf(pdf_path)
    collection_id = "metadata-policy-id"
    container = get_container()
    container.ingest_document_use_case.execute(source_path=str(pdf_path), collection_id=collection_id)

    catalog = build_chunk_catalog(container.index_registry, collection_id)
    policy_ids = {metadata.get("policy_id") for _, _, metadata in catalog if metadata.get("policy_id")}

    assert "HR-203" in policy_ids
    assert "HR-105" in policy_ids
