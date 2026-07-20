"""Retrieval overlap utilities."""

from __future__ import annotations

from adaptive_rag.domain.models.retrieval import ScoredChunk


def compute_retrieval_overlap(
    dense_hits: list[ScoredChunk],
    sparse_hits: list[ScoredChunk],
    *,
    top_k: int = 10,
) -> float:
    """Return overlap ratio between dense and sparse top-k chunk ids."""
    dense_ids = {hit.chunk.id for hit in dense_hits[:top_k]}
    sparse_ids = {hit.chunk.id for hit in sparse_hits[:top_k]}
    if not dense_ids and not sparse_ids:
        return 0.0
    union = dense_ids | sparse_ids
    if not union:
        return 0.0
    return len(dense_ids & sparse_ids) / len(union)
