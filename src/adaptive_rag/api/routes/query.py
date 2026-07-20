"""Query and ingestion routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from adaptive_rag.api.dependencies.container import Container, get_container
from adaptive_rag.api.schemas.query import QueryRequest, QueryResponse

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
