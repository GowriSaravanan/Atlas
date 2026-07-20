"""Confidence scoring domain models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ConfidenceBreakdown(BaseModel):
    """Explainable sub-scores that compose retrieval confidence."""

    reranker_score: float = 0.0
    reranker_margin: float = 0.0
    retrieval_overlap: float = 0.0
    metadata_match: float = 0.0
    evidence_density: float = 0.0
    citation_coverage: float | None = None


class ConfidenceScore(BaseModel):
    """Composite confidence with auditable breakdown and weights."""

    value: float
    is_acceptable: bool
    threshold: float
    breakdown: ConfidenceBreakdown = Field(default_factory=ConfidenceBreakdown)
    weights: dict[str, float] = Field(default_factory=dict)
