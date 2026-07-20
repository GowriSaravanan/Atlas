"""API request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

from adaptive_rag.application.dto.responses import RAGResponse


class QueryRequest(BaseModel):
    """HTTP request body for RAG query."""

    query: str = Field(min_length=1)
    conversation_id: str = "default"


class QueryResponse(RAGResponse):
    """HTTP response for RAG query — extends application DTO."""

    pass


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    app_name: str
    version: str
