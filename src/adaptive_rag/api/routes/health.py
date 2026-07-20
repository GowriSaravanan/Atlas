"""Health check routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from adaptive_rag.api.dependencies.container import Container, get_container
from adaptive_rag.api.schemas.query import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check(container: Container = Depends(get_container)) -> HealthResponse:
    """Return service health status."""
    settings = container.settings
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=settings.app_version,
    )
