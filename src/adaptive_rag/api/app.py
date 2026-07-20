"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from adaptive_rag.api.dependencies.container import get_container
from adaptive_rag.api.routes import health, query
from adaptive_rag.config.settings import get_settings
from adaptive_rag.observability.logging import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan — initialize logging and DI container."""
    settings = get_settings()
    setup_logging(settings.logging)
    container = get_container()
    logger.info(
        "Starting %s v%s",
        settings.app_name,
        settings.app_version,
        extra={"ctx_debug": settings.debug},
    )
    yield
    logger.info("Shutting down %s", settings.app_name)
    _ = container  # container teardown hooks in later phases


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(query.router, prefix="/api/v1")
    return app


app = create_app()
