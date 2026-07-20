"""Observability and retrieval trace models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from adaptive_rag.domain.models.confidence import ConfidenceBreakdown, ConfidenceScore
from adaptive_rag.domain.models.grounding import CitationReport, GroundingReport, GuardDecision
from adaptive_rag.domain.models.query import OriginalQueryAnalysis, QueryAnalysis, ResolvedQueryAnalysis
from adaptive_rag.domain.models.retrieval import RetrievalStrategy, ScoredChunk, SearchScope


class StepTrace(BaseModel):
    """Trace record for a single pipeline step."""

    step: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    duration_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalTrace(BaseModel):
    """Full explainability trace for a query execution."""

    query_id: str = ""
    raw_query: str = ""
    resolved_query: str | None = None
    original_analysis: OriginalQueryAnalysis | None = None
    resolved_analysis: ResolvedQueryAnalysis | None = None
    search_scope: SearchScope | None = None
    strategy: RetrievalStrategy | None = None
    dense_hits: list[ScoredChunk] = Field(default_factory=list)
    sparse_hits: list[ScoredChunk] = Field(default_factory=list)
    fused_hits: list[ScoredChunk] = Field(default_factory=list)
    reranked_hits: list[ScoredChunk] = Field(default_factory=list)
    retrieval_confidence: ConfidenceScore | None = None
    final_confidence: ConfidenceScore | None = None
    confidence_breakdown: ConfidenceBreakdown | None = None
    grounding_report: GroundingReport | None = None
    citation_report: CitationReport | None = None
    guard_decision: GuardDecision | None = None
    escalation_count: int = 0
    steps: list[StepTrace] = Field(default_factory=list)
    latency_ms: dict[str, float] = Field(default_factory=dict)
