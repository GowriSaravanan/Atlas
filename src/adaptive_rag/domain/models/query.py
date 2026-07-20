"""Query analysis domain models."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from adaptive_rag.domain.models.retrieval import RetrievalStrategy


class QueryType(str, Enum):
    """Stable query taxonomy used across routing, rewriting, and decomposition."""

    LOOKUP = "lookup"
    FACTUAL = "factual"
    SEMANTIC = "semantic"
    COMPARISON = "comparison"
    MULTI_HOP = "multi_hop"
    AMBIGUOUS = "ambiguous"
    CONVERSATIONAL = "conversational"


class QueryIntent(str, Enum):
    """High-level intent classification for a user query."""

    LOOKUP = "lookup"
    FACTUAL = "factual"
    COMPARATIVE = "comparative"
    PROCEDURAL = "procedural"
    SUMMARIZATION = "summarization"
    CHITCHAT = "chitchat"
    UNKNOWN = "unknown"


class ComplexityLevel(str, Enum):
    """Estimated complexity of a user query."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RewriteDecision(BaseModel):
    """Explainable decision for pre-retrieval query rewriting."""

    should_rewrite: bool = False
    reason: str = ""


class RewriteResult(BaseModel):
    """Outcome of a query rewrite attempt."""

    original_query: str
    rewritten_query: str
    was_rewritten: bool = False
    reason: str = ""


class DecompositionDecision(BaseModel):
    """Explainable decision for query decomposition."""

    should_decompose: bool = False
    reason: str = ""


class QueryAnalysis(BaseModel):
    """Structured output from query analysis (rules + LLM)."""

    query_type: QueryType = QueryType.FACTUAL
    intent: QueryIntent = QueryIntent.UNKNOWN
    complexity: ComplexityLevel = ComplexityLevel.MEDIUM
    entities: list[str] = Field(default_factory=list)
    metadata_hints: dict[str, Any] = Field(default_factory=dict)
    metadata_scope: dict[str, Any] = Field(default_factory=dict)
    is_multi_question: bool = False
    rewrite_decision: RewriteDecision = Field(default_factory=RewriteDecision)
    decomposition_decision: DecompositionDecision = Field(default_factory=DecompositionDecision)
    rule_matches: list[str] = Field(default_factory=list)

    # Legacy flags — derived from decisions for backward compatibility
    needs_pre_rewrite: bool = False
    needs_decomposition: bool = False


# Semantic aliases — same structure, distinct roles in traces and observability.
OriginalQueryAnalysis = QueryAnalysis
ResolvedQueryAnalysis = QueryAnalysis
EffectiveQueryAnalysis = ResolvedQueryAnalysis


class RetrievalDecision(BaseModel):
    """Routing decision produced by the adaptive router."""

    strategy: RetrievalStrategy
    reason: str
