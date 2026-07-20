"""Reranker port."""

from __future__ import annotations

from typing import Protocol

from adaptive_rag.domain.models.retrieval import ScoredChunk


class RerankerPort(Protocol):
    """Cross-encoder reranking."""

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int,
    ) -> list[ScoredChunk]:
        """Rerank candidate chunks against the query."""
        ...
