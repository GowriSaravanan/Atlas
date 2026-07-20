"""Retrieval domain models."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from adaptive_rag.domain.models.confidence import ConfidenceScore
from adaptive_rag.domain.models.document import Chunk


class RetrievalStrategy(str, Enum):
    """Retrieval strategy selected by the adaptive router."""

    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"


class SearchScope(BaseModel):
    """Scoped search space after metadata filtering."""

    filters: dict[str, Any] = Field(default_factory=dict)
    estimated_corpus_size: int = 0
    scope_metadata: dict[str, Any] = Field(default_factory=dict)


class ScoredChunk(BaseModel):
    """A chunk with an associated retrieval or reranking score."""

    chunk: Chunk
    score: float
    source: str = "unknown"
    rank: int | None = None


class SubqueryState(BaseModel):
    """State tracked for an individual decomposed subquery."""

    index: int
    text: str
    strategy: RetrievalStrategy | None = None
    metadata_filters: dict[str, Any] = Field(default_factory=dict)
    retrieval_sets: list[ScoredChunk] = Field(default_factory=list)
    confidence: ConfidenceScore | None = None
    escalation_level: int = 0
    status: Literal["pending", "ok", "retry", "failed"] = "pending"


class SubquerySummary(BaseModel):
    """Summary of a subquery execution for the response payload."""

    index: int
    text: str
    strategy: RetrievalStrategy | None
    status: Literal["pending", "ok", "retry", "failed"]
    confidence: ConfidenceScore | None = None
