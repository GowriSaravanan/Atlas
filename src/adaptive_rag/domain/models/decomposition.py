"""Query decomposition domain models."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from adaptive_rag.domain.models.confidence import ConfidenceScore
from adaptive_rag.domain.models.query import QueryType
from adaptive_rag.domain.models.retrieval import RetrievalStrategy, ScoredChunk
from adaptive_rag.domain.models.trace import RetrievalTrace


class SubQuerySource(str, Enum):
    """Origin of a decomposed subquery."""

    PASS_THROUGH = "pass_through"
    COMPARISON = "comparison"
    MULTI_HOP = "multi_hop"
    MULTI_QUESTION = "multi_question"


class SubQuery(BaseModel):
    """Standalone retrieval unit produced by decomposition."""

    id: str
    query: str
    entity: str | None = None
    source: SubQuerySource
    query_type: QueryType
    parent_query: str


class DecompositionResult(BaseModel):
    """Outcome of query decomposition — always contains at least one subquery."""

    original_query: str
    was_decomposed: bool = False
    reason: str = ""
    subqueries: list[SubQuery] = Field(default_factory=list)


class SubQueryRetrievalPlan(BaseModel):
    """Per-subquery routing and budget decision."""

    subquery: SubQuery
    strategy: RetrievalStrategy
    reason: str
    top_k: int


class SubQueryResult(BaseModel):
    """Full outcome of retrieving a single subquery."""

    subquery: SubQuery
    plan: SubQueryRetrievalPlan
    results: list[ScoredChunk] = Field(default_factory=list)
    dense_hits: list[ScoredChunk] = Field(default_factory=list)
    sparse_hits: list[ScoredChunk] = Field(default_factory=list)
    fused_hits: list[ScoredChunk] = Field(default_factory=list)
    retrieval_overlap: float = 0.0
    confidence: ConfidenceScore
    trace: RetrievalTrace = Field(default_factory=RetrievalTrace)
