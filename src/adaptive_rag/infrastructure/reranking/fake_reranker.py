"""Deterministic passthrough reranker for tests and eval."""

from __future__ import annotations

from adaptive_rag.domain.models.retrieval import ScoredChunk
from adaptive_rag.domain.ports.reranker import RerankerPort


class FakeReranker(RerankerPort):
    """Preserve retrieval order while tagging results as reranked."""

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int,
    ) -> list[ScoredChunk]:
        del query
        if not candidates:
            return []

        limit = max(1, min(top_k, len(candidates)))
        return [
            ScoredChunk(
                chunk=candidate.chunk,
                score=candidate.score,
                source="reranker",
                rank=rank,
            )
            for rank, candidate in enumerate(candidates[:limit], start=1)
        ]
