"""Collection and ingestion routes."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, UploadFile

from adaptive_rag.api.dependencies.container import Container, get_container
from adaptive_rag.application.dto.responses import IngestDocumentRequest, IngestDocumentResponse
from adaptive_rag.application.security.paths import resolve_path_within_directory, validate_pdf_upload
from adaptive_rag.domain.models.index import CollectionStats
from adaptive_rag.domain.validation.collection_id import validate_collection_id

router = APIRouter(tags=["ingestion"])


@router.post("/ingest", response_model=IngestDocumentResponse)
def ingest(
    request: IngestDocumentRequest,
    container: Container = Depends(get_container),
) -> IngestDocumentResponse:
    """Ingest a document from a path inside the configured upload directory."""
    container.ensure_storage_dirs()
    upload_dir = Path(container.settings.storage.upload_dir)
    source_path = resolve_path_within_directory(request.source_path, upload_dir)
    validate_collection_id(request.collection_id)
    return container.ingest_document_use_case.execute(
        source_path=str(source_path),
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
    validate_collection_id(collection_id)
    upload_dir = Path(container.settings.storage.upload_dir)
    max_bytes = container.settings.storage.max_upload_bytes

    if file.content_type not in {None, "application/pdf", "application/x-pdf"}:
        raise ValueError("Upload must use content type application/pdf")

    content = await file.read()
    if len(content) > max_bytes:
        await file.close()
        raise ValueError(f"Uploaded file exceeds maximum size of {max_bytes} bytes")
    validate_pdf_upload(content, max_bytes=max_bytes)

    suffix = Path(file.filename or "document.pdf").suffix or ".pdf"
    if suffix.lower() != ".pdf":
        raise ValueError("Upload filename must use a .pdf extension")

    destination = upload_dir / f"{uuid.uuid4()}{suffix}"
    destination.write_bytes(content)
    await file.close()

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
    validate_collection_id(collection_id)
    return container.index_registry.stats(collection_id)
