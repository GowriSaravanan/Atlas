"""Plain configuration objects injected into domain policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RetrievalPolicyConfig:
    """Retrieval confidence thresholds used by domain policies."""

    confidence_threshold: float


@dataclass(frozen=True)
class ConfidenceWeightConfig:
    """Weights for composite confidence scoring."""

    reranker_score: float
    reranker_margin: float
    retrieval_overlap: float
    metadata_match: float
    evidence_density: float


@dataclass(frozen=True)
class FusionPolicyConfig:
    """Reciprocal rank fusion parameters."""

    rrf_k: int


@dataclass(frozen=True)
class ChunkingPolicyConfig:
    """Adaptive chunking parameters."""

    max_tokens: int
    min_tokens: int
    overlap_tokens: int
    strategy: Literal["structure_aware", "fixed"]


@dataclass(frozen=True)
class AnswerGenerationPolicyConfig:
    """Answer-generation context assembly parameters."""

    max_context_tokens: int


@dataclass(frozen=True)
class CitationFormatterPolicyConfig:
    """Citation formatting parameters."""

    excerpt_max_chars: int
