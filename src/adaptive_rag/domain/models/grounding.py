"""Grounding and citation verification domain models."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from adaptive_rag.domain.models.citation import Citation


class GuardDecision(str, Enum):
    """Action decided by the hallucination guard."""

    PASS = "pass"
    REGENERATE = "regenerate"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"


class ClaimSupport(BaseModel):
    """Mapping between a factual claim and supporting evidence."""

    claim: str
    is_supported: bool
    supporting_chunk_ids: list[str] = Field(default_factory=list)


class GroundingReport(BaseModel):
    """Result of validating answer claims against retrieved evidence."""

    claims: list[ClaimSupport] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    all_grounded: bool = False


class CitationValidation(BaseModel):
    """Validation result for a single citation."""

    citation_index: int
    chunk_id: str
    is_valid: bool
    reason: str = ""


class CitationReport(BaseModel):
    """Result of verifying citation-to-chunk correctness."""

    validations: list[CitationValidation] = Field(default_factory=list)
    all_valid: bool = False


class AnswerMode(str, Enum):
    """Indicates completeness of the generated answer."""

    FULL = "full"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"


class GuardResult(BaseModel):
    """Combined output from the hallucination guard."""

    decision: GuardDecision
    answer_mode: Literal["full", "partial", "insufficient"]
    grounding_report: GroundingReport
    citation_report: CitationReport
    message: str = ""
