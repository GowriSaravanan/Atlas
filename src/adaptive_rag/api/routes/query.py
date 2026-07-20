"""Query and ingestion routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from adaptive_rag.api.dependencies.container import Container, get_container
from adaptive_rag.api.schemas.query import QueryRequest, QueryResponse
from adaptive_rag.application.dto.responses import IngestDocumentRequest, IngestDocumentResponse

router = APIRouter(tags=["rag"])


@router.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    container: Container = Depends(get_container),
) -> QueryResponse:
    """Execute a RAG query through the workflow graph."""
    result = container.query_rag_use_case.execute(
        query=request.query,
        conversation_id=request.conversation_id,
    )
    return QueryResponse.model_validate(result.model_dump())


@router.post("/ingest", response_model=IngestDocumentResponse)
def ingest(
    request: IngestDocumentRequest,
    container: Container = Depends(get_container),
) -> IngestDocumentResponse:
    """Ingest a document through the ingestion pipeline."""
    return container.ingest_document_use_case.execute(
        source_path=request.source_path,
        collection_id=request.collection_id,
    )
