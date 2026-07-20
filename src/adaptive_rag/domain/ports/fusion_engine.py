"""Fusion engine port."""

from __future__ import annotations

from typing import Protocol

from adaptive_rag.domain.models.retrieval import ScoredChunk


class FusionEnginePort(Protocol):
    """Fuse multiple ranked retrieval lists into one."""

    def fuse(
        self,
        ranked_lists: dict[str, list[ScoredChunk]],
        query: str,
    ) -> list[ScoredChunk]:
        """Fuse named ranked lists (e.g. 'bm25', 'dense') into a single list."""
        ...
