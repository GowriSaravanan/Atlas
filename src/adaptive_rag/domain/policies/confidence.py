"""Retrieval confidence scoring policy."""

from __future__ import annotations

from adaptive_rag.domain.config.policy_config import ConfidenceWeightConfig, RetrievalPolicyConfig
from adaptive_rag.domain.models.confidence import ConfidenceBreakdown, ConfidenceScore
from adaptive_rag.domain.models.query import QueryAnalysis
from adaptive_rag.domain.models.retrieval import ScoredChunk


class ConfidenceScorer:
    """Compute explainable retrieval confidence from hybrid results."""

    def __init__(
        self,
        *,
        retrieval: RetrievalPolicyConfig,
        weights: ConfidenceWeightConfig,
    ) -> None:
        self._threshold = retrieval.confidence_threshold
        self._weights = {
            "retrieval_overlap": weights.retrieval_overlap,
            "reranker_score": weights.reranker_score,
            "metadata_match": weights.metadata_match,
        }

    def score(
        self,
        *,
        results: list[ScoredChunk],
        retrieval_overlap: float,
        analysis: QueryAnalysis,
    ) -> ConfidenceScore:
        """Score retrieval quality using overlap, top hit score, and metadata alignment."""
        top_score = self._normalize_top_score(results)
        metadata_match = self._metadata_match_score(results, analysis)

        breakdown = ConfidenceBreakdown(
            reranker_score=top_score,
            reranker_margin=self._score_margin(results),
            retrieval_overlap=retrieval_overlap,
            metadata_match=metadata_match,
            evidence_density=min(len(results) / 5.0, 1.0),
        )

        value = (
            self._weights["retrieval_overlap"] * breakdown.retrieval_overlap
            + self._weights["reranker_score"] * breakdown.reranker_score
            + self._weights["metadata_match"] * breakdown.metadata_match
        )

        return ConfidenceScore(
            value=round(value, 4),
            is_acceptable=value >= self._threshold,
            threshold=self._threshold,
            breakdown=breakdown,
            weights=dict(self._weights),
        )

    def aggregate(self, scores: list[ConfidenceScore]) -> ConfidenceScore | None:
        """Average subquery confidence scores into a composite score."""
        if not scores:
            return None

        avg_value = sum(score.value for score in scores) / len(scores)
        avg_overlap = sum(score.breakdown.retrieval_overlap for score in scores) / len(scores)
        avg_metadata = sum(score.breakdown.metadata_match for score in scores) / len(scores)
        avg_top = sum(score.breakdown.reranker_score for score in scores) / len(scores)

        return ConfidenceScore(
            value=round(avg_value, 4),
            is_acceptable=avg_value >= self._threshold,
            threshold=self._threshold,
            breakdown=ConfidenceBreakdown(
                reranker_score=avg_top,
                reranker_margin=sum(score.breakdown.reranker_margin for score in scores) / len(scores),
                retrieval_overlap=avg_overlap,
                metadata_match=avg_metadata,
                evidence_density=sum(score.breakdown.evidence_density for score in scores) / len(scores),
            ),
            weights=dict(self._weights),
        )

    @staticmethod
    def _normalize_top_score(results: list[ScoredChunk]) -> float:
        if not results:
            return 0.0
        raw = results[0].score
        if raw <= 0:
            return 0.0
        if raw <= 1.0:
            return raw
        return min(raw / (raw + 1.0), 1.0)

    @staticmethod
    def _score_margin(results: list[ScoredChunk]) -> float:
        if len(results) < 2:
            return 0.0
        top = ConfidenceScorer._normalize_top_score(results)
        second = ConfidenceScorer._normalize_top_score([results[1]])
        return max(top - second, 0.0)

    @staticmethod
    def _metadata_match_score(results: list[ScoredChunk], analysis: QueryAnalysis) -> float:
        if not analysis.metadata_scope or not results:
            return 0.0 if analysis.metadata_scope else 1.0

        top_chunk = results[0].chunk
        matched = 0
        for key, expected in analysis.metadata_scope.items():
            actual = top_chunk.metadata.get(key)
            if actual is None:
                continue
            if str(actual).lower() == str(expected).lower():
                matched += 1

        return matched / len(analysis.metadata_scope)
