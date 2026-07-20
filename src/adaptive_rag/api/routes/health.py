"""Health check routes."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Depends

from adaptive_rag.api.dependencies.container import Container, get_container
from adaptive_rag.api.schemas.query import HealthResponse, ReadinessResponse

router = APIRouter(tags=["health"])


def _llm_is_configured(settings) -> bool:
    llm = settings.llm
    if not llm.model:
        return False
    if llm.provider == "openrouter":
        return bool(llm.openrouter_api_key)
    if llm.provider == "openai":
        return bool(llm.openai_api_key)
    if llm.provider == "gemini":
        return bool(llm.gemini_api_key)
    if llm.provider == "groq":
        return bool(llm.groq_api_key)
    return True


@router.get("/health", response_model=HealthResponse)
def health_check(container: Container = Depends(get_container)) -> HealthResponse:
    """Return lightweight service health status."""
    settings = container.settings
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=settings.app_version,
    )


@router.get("/ready", response_model=ReadinessResponse)
def readiness_check(container: Container = Depends(get_container)) -> ReadinessResponse:
    """Verify critical dependencies before accepting production traffic."""
    settings = container.settings
    container.ensure_storage_dirs()

    index_dir = Path(settings.storage.index_dir)
    upload_dir = Path(settings.storage.upload_dir)
    checks = {
        "index_dir_writable": index_dir.exists() and os.access(index_dir, os.W_OK),
        "upload_dir_writable": upload_dir.exists() and os.access(upload_dir, os.W_OK),
        "embedder_initialized": container.embedder.dimension > 0,
        "llm_configured": _llm_is_configured(settings),
    }
    return ReadinessResponse(ready=all(checks.values()), checks=checks)
