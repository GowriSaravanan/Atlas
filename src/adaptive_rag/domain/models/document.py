"""Document and chunk domain models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A source document loaded from ingestion."""

    id: str
    source_path: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Chunk(BaseModel):
    """A text chunk derived from a document."""

    id: str
    document_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    token_count: int = 0
    chunk_index: int = 0
