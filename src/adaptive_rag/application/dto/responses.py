"""Application DTOs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from adaptive_rag.domain.models.confidence import ConfidenceScore
from adaptive_rag.domain.models.grounding import Citation
from adaptive_rag.domain.models.retrieval import RetrievalStrategy, SubquerySummary
from adaptive_rag.domain.models.trace import RetrievalTrace


class RAGResponse(BaseModel):
    """Public response payload for a RAG query."""

    answer: str
    answer_mode: Literal["full", "partial", "insufficient"]
    citations: list[Citation] = Field(default_factory=list)
    global_confidence: ConfidenceScore | None = None
    strategy_used: RetrievalStrategy | None = None
    subquery_summaries: list[SubquerySummary] = Field(default_factory=list)
    escalation_count: int = 0
    trace: RetrievalTrace = Field(default_factory=RetrievalTrace)


class IngestDocumentRequest(BaseModel):
    """Request to ingest a document."""

    source_path: str
    collection_id: str = "default"


class IngestDocumentResponse(BaseModel):
    """Response after document ingestion."""

    document_id: str
    chunk_count: int
    status: Literal["accepted", "completed", "failed"] = "accepted"
    message: str = ""
