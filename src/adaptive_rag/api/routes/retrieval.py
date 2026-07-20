"""Retrieval routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from adaptive_rag.api.dependencies.container import Container, get_container
from adaptive_rag.application.dto.retrieval import HybridRetrievalResult, RetrievalEngineResult, RetrievalRequest

router = APIRouter(tags=["retrieval"])


@router.post("/retrieve", response_model=RetrievalEngineResult)
def retrieve(
    request: RetrievalRequest,
    container: Container = Depends(get_container),
) -> RetrievalEngineResult:
    """Run adaptive hybrid retrieval with query analysis and confidence scoring."""
    return container.hybrid_retrieval_use_case.execute_request(request)
