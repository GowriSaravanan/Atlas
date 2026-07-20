"""Reciprocal Rank Fusion implementation."""

from __future__ import annotations

from adaptive_rag.domain.config.policy_config import FusionPolicyConfig
from adaptive_rag.domain.models.retrieval import ScoredChunk
from adaptive_rag.domain.ports.fusion_engine import FusionEnginePort


class ReciprocalRankFusion(FusionEnginePort):
    """Fuse multiple ranked lists using Reciprocal Rank Fusion (RRF)."""

    def __init__(self, config: FusionPolicyConfig) -> None:
        self._k = config.rrf_k

    def fuse(
        self,
        ranked_lists: dict[str, list[ScoredChunk]],
        query: str,
    ) -> list[ScoredChunk]:
        """Fuse named ranked lists into a single ranked list."""
        _ = query  # reserved for weighted / query-aware fusion variants
        if not ranked_lists:
            return []

        if len(ranked_lists) == 1:
            return next(iter(ranked_lists.values()))

        fused_scores: dict[str, float] = {}
        chunk_lookup: dict[str, ScoredChunk] = {}

        for source, ranked_list in ranked_lists.items():
            for rank, scored_chunk in enumerate(ranked_list, start=1):
                chunk_id = scored_chunk.chunk.id
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (
                    1.0 / (self._k + rank)
                )
                chunk_lookup.setdefault(chunk_id, scored_chunk)

        sorted_ids = sorted(fused_scores, key=lambda chunk_id: fused_scores[chunk_id], reverse=True)
        return [
            ScoredChunk(
                chunk=chunk_lookup[chunk_id].chunk,
                score=fused_scores[chunk_id],
                source="rrf",
                rank=index + 1,
            )
            for index, chunk_id in enumerate(sorted_ids)
        ]
