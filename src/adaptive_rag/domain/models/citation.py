"""Citation formatting domain models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Structured evidence reference linked to a retrieved chunk."""

    chunk_id: str
    document_id: str
    page_number: int | None = None
    section_title: str = ""
    confidence: float | None = None
    excerpt: str = ""
    claim: str = ""


class CitationFormats(BaseModel):
    """Rendered answer views with evidence attribution."""

    markdown: str
    plain_text: str
    json_text: str
