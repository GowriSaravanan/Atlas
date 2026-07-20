"""Collection and ingestion routes."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, UploadFile

from adaptive_rag.api.dependencies.container import Container, get_container
from adaptive_rag.application.dto.responses import IngestDocumentRequest, IngestDocumentResponse
from adaptive_rag.domain.models.index import CollectionStats

router = APIRouter(tags=["ingestion"])


@router.post("/ingest", response_model=IngestDocumentResponse)
def ingest(
    request: IngestDocumentRequest,
    container: Container = Depends(get_container),
) -> IngestDocumentResponse:
    """Ingest a document from a local file path."""
    return container.ingest_document_use_case.execute(
        source_path=request.source_path,
        collection_id=request.collection_id,
    )


@router.post("/ingest/upload", response_model=IngestDocumentResponse)
async def ingest_upload(
    file: UploadFile = File(...),
    collection_id: str = Form(default="default"),
    container: Container = Depends(get_container),
) -> IngestDocumentResponse:
    """Upload and ingest a PDF document."""
    container.ensure_storage_dirs()
    upload_dir = Path(container.settings.storage.upload_dir)
    suffix = Path(file.filename or "document.pdf").suffix or ".pdf"
    destination = upload_dir / f"{uuid.uuid4()}{suffix}"

    content = await file.read()
    destination.write_bytes(content)

    return container.ingest_document_use_case.execute(
        source_path=str(destination),
        collection_id=collection_id,
    )


@router.get("/collections/{collection_id}/stats", response_model=CollectionStats)
def collection_stats(
    collection_id: str,
    container: Container = Depends(get_container),
) -> CollectionStats:
    """Return indexing statistics for a collection."""
    return container.index_registry.stats(collection_id)
