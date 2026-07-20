"""Query decomposer port."""

from __future__ import annotations

from typing import Protocol

from adaptive_rag.domain.models.decomposition import DecompositionResult
from adaptive_rag.domain.models.query import ResolvedQueryAnalysis


class QueryDecomposerPort(Protocol):
    """Decompose a resolved query into standalone subqueries."""

    def decompose(
        self,
        query: str,
        analysis: ResolvedQueryAnalysis,
    ) -> DecompositionResult:
        """Return decomposition result with at least one subquery."""
        ...
