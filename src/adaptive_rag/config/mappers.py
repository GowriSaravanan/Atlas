"""Map application settings to domain configuration value objects."""

from __future__ import annotations

from pathlib import Path

from adaptive_rag.config.settings import (
    AnswerGenerationSettings,
    ChunkingSettings,
    ConfidenceWeightSettings,
    RetrievalSettings,
    Settings,
)
from adaptive_rag.domain.config.policy_config import (
    AnswerGenerationPolicyConfig,
    ChunkingPolicyConfig,
    ConfidenceWeightConfig,
    FusionPolicyConfig,
    RetrievalPolicyConfig,
)


def to_retrieval_policy_config(settings: RetrievalSettings) -> RetrievalPolicyConfig:
    return RetrievalPolicyConfig(confidence_threshold=settings.confidence_threshold)


def to_confidence_weight_config(settings: ConfidenceWeightSettings) -> ConfidenceWeightConfig:
    return ConfidenceWeightConfig(
        reranker_score=settings.reranker_score,
        reranker_margin=settings.reranker_margin,
        retrieval_overlap=settings.retrieval_overlap,
        metadata_match=settings.metadata_match,
        evidence_density=settings.evidence_density,
    )


def to_fusion_policy_config(settings: RetrievalSettings) -> FusionPolicyConfig:
    return FusionPolicyConfig(rrf_k=settings.rrf_k)


def to_chunking_policy_config(settings: ChunkingSettings) -> ChunkingPolicyConfig:
    return ChunkingPolicyConfig(
        max_tokens=settings.max_tokens,
        min_tokens=settings.min_tokens,
        overlap_tokens=settings.overlap_tokens,
        strategy=settings.strategy,
    )


def to_answer_generation_policy_config(
    settings: AnswerGenerationSettings,
) -> AnswerGenerationPolicyConfig:
    return AnswerGenerationPolicyConfig(max_context_tokens=settings.max_context_tokens)


def resolve_prompts_dir(settings: Settings) -> Path:
    """Resolve prompt template directory relative to repository root."""
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / settings.answer_generation.prompts_dir
