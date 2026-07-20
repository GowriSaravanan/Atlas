"""Merge ranked results from multiple subqueries."""

from __future__ import annotations

from adaptive_rag.domain.models.decomposition import SubQueryResult
from adaptive_rag.domain.models.retrieval import ScoredChunk
from adaptive_rag.domain.ports.fusion_engine import FusionEnginePort


class SubqueryResultMerger:
    """Fuse per-subquery ranked lists into a single ranked list."""

    def __init__(self, fusion_engine: FusionEnginePort) -> None:
        self._fusion_engine = fusion_engine

    def merge(
        self,
        subquery_results: list[SubQueryResult],
        *,
        top_k: int,
        parent_query: str,
    ) -> list[ScoredChunk]:
        """Merge subquery results with RRF and return the top global hits."""
        ranked_lists = {
            f"subquery_{result.subquery.id}": result.results
            for result in subquery_results
            if result.results
        }
        if not ranked_lists:
            return []
        if len(ranked_lists) == 1:
            return next(iter(ranked_lists.values()))[:top_k]

        merged = self._fusion_engine.fuse(ranked_lists, parent_query)
        return merged[:top_k]
