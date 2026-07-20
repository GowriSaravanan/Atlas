"""Retrieval DTOs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from adaptive_rag.domain.models.conversation import Message
from adaptive_rag.domain.models.confidence import ConfidenceScore
from adaptive_rag.domain.models.answer import GeneratedAnswer
from adaptive_rag.domain.models.decomposition import DecompositionResult, SubQueryResult
from adaptive_rag.domain.models.query import (
    OriginalQueryAnalysis,
    QueryAnalysis,
    ResolvedQueryAnalysis,
    RetrievalDecision,
    RewriteResult,
)
from adaptive_rag.domain.models.retrieval import RetrievalStrategy, ScoredChunk
from adaptive_rag.domain.models.trace import RetrievalTrace


class RetrievalRequest(BaseModel):
    """Request to run adaptive hybrid retrieval."""

    query: str = Field(min_length=1)
    collection_id: str = "default"
    conversation_id: str | None = None
    context_messages: list[Message] = Field(default_factory=list)
    strategy: RetrievalStrategy | None = None
    metadata_filters: dict[str, Any] = Field(default_factory=dict)
    top_k: int | None = None


class HybridRetrievalResult(BaseModel):
    """Result of a hybrid retrieval operation (retrieval only)."""

    query: str
    collection_id: str
    strategy: RetrievalStrategy
    results: list[ScoredChunk] = Field(default_factory=list)
    dense_hits: list[ScoredChunk] = Field(default_factory=list)
    sparse_hits: list[ScoredChunk] = Field(default_factory=list)
    fused_hits: list[ScoredChunk] = Field(default_factory=list)
    retrieval_overlap: float = 0.0
    trace: RetrievalTrace = Field(default_factory=RetrievalTrace)


class RetrievalEngineResult(HybridRetrievalResult):
    """Adaptive retrieval result including analysis, routing, and confidence."""

    resolved_query: str | None = None
    rewrite_result: RewriteResult | None = None
    original_analysis: OriginalQueryAnalysis | None = None
    resolved_analysis: ResolvedQueryAnalysis | None = None
    decomposition_result: DecompositionResult | None = None
    subquery_results: list[SubQueryResult] = Field(default_factory=list)
    decision: RetrievalDecision | None = None
    confidence: ConfidenceScore | None = None
    generated_answer: GeneratedAnswer | None = None

    @property
    def analysis(self) -> ResolvedQueryAnalysis | None:
        """Effective analysis used for routing and retrieval (alias for resolved_analysis)."""
        return self.resolved_analysis
