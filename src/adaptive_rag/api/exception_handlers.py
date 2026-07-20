"""Structured API error responses and exception handlers."""

from __future__ import annotations

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from adaptive_rag.domain.errors import (
    CollectionNotFoundError,
    DomainError,
    EmbedderCompatibilityError,
    InvalidCollectionIdError,
    PathAccessError,
    ProviderError,
)
from adaptive_rag.observability.logging import get_logger

logger = get_logger(__name__)


class ErrorResponse(BaseModel):
    """Consistent JSON error payload."""

    error: str
    code: str
    detail: str | None = None


def _error_response(
    *,
    status_code: int,
    code: str,
    error: str,
    detail: str | None = None,
) -> JSONResponse:
    payload = ErrorResponse(error=error, code=code, detail=detail)
    return JSONResponse(status_code=status_code, content=payload.model_dump())


def register_exception_handlers(app: FastAPI) -> None:
    """Register centralized exception handlers."""

    @app.exception_handler(InvalidCollectionIdError)
    async def invalid_collection_id_handler(
        _request: Request,
        exc: InvalidCollectionIdError,
    ) -> JSONResponse:
        return _error_response(status_code=422, code=exc.code, error=exc.message)

    @app.exception_handler(CollectionNotFoundError)
    async def collection_not_found_handler(
        _request: Request,
        exc: CollectionNotFoundError,
    ) -> JSONResponse:
        return _error_response(status_code=404, code=exc.code, error=exc.message)

    @app.exception_handler(PathAccessError)
    async def path_access_handler(_request: Request, exc: PathAccessError) -> JSONResponse:
        return _error_response(status_code=403, code=exc.code, error=exc.message)

    @app.exception_handler(EmbedderCompatibilityError)
    async def embedder_handler(
        _request: Request,
        exc: EmbedderCompatibilityError,
    ) -> JSONResponse:
        return _error_response(status_code=409, code=exc.code, error=exc.message)

    @app.exception_handler(ProviderError)
    async def provider_handler(_request: Request, exc: ProviderError) -> JSONResponse:
        return _error_response(status_code=502, code=exc.code, error=exc.message)

    @app.exception_handler(DomainError)
    async def domain_error_handler(_request: Request, exc: DomainError) -> JSONResponse:
        return _error_response(status_code=400, code=exc.code, error=exc.message)

    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(_request: Request, exc: FileNotFoundError) -> JSONResponse:
        return _error_response(status_code=404, code="not_found", error=str(exc))

    @app.exception_handler(ValueError)
    async def value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
        return _error_response(status_code=422, code="validation_error", error=str(exc))

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return _error_response(
            status_code=422,
            code="request_validation_error",
            error="Request validation failed",
            detail=str(exc.errors()),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        _request: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        return _error_response(status_code=exc.status_code, code="http_error", error=detail)

    @app.exception_handler(httpx.HTTPError)
    async def httpx_error_handler(_request: Request, exc: httpx.HTTPError) -> JSONResponse:
        logger.warning("External provider request failed", extra={"ctx_error": str(exc)})
        return _error_response(
            status_code=502,
            code="provider_error",
            error="External provider request failed",
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled server error", exc_info=exc)
        return _error_response(
            status_code=500,
            code="internal_error",
            error="An internal server error occurred",
        )
