"""Scope filtering utilities for retrieval."""

from __future__ import annotations

from adaptive_rag.domain.models.document import Chunk
from adaptive_rag.domain.models.retrieval import SearchScope


def chunk_matches_scope(chunk: Chunk, scope: SearchScope) -> bool:
    """Return True when a chunk satisfies all scope filters."""
    if not scope.filters:
        return True

    for key, expected in scope.filters.items():
        actual = chunk.metadata.get(key)
        if actual is None:
            return False
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def count_scope_matches(chunks: list[Chunk], scope: SearchScope) -> int:
    """Count chunks matching the given scope."""
    return sum(1 for chunk in chunks if chunk_matches_scope(chunk, scope))
