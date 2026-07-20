"""Weighted top-k budget allocation for subqueries."""

from __future__ import annotations

from adaptive_rag.domain.models.decomposition import SubQuery
from adaptive_rag.domain.models.query import QueryType


class TopKAllocator:
    """Allocate retrieval budget across subqueries based on query type."""

    _TYPE_WEIGHTS: dict[QueryType, int] = {
        QueryType.LOOKUP: 3,
        QueryType.FACTUAL: 5,
        QueryType.SEMANTIC: 7,
        QueryType.COMPARISON: 5,
        QueryType.MULTI_HOP: 5,
        QueryType.AMBIGUOUS: 5,
        QueryType.CONVERSATIONAL: 3,
    }

    def allocate(self, subqueries: list[SubQuery], total_budget: int) -> dict[str, int]:
        """Return a subquery-id to top-k mapping that sums to at most total_budget."""
        if not subqueries:
            return {}

        budget = max(1, total_budget)
        if len(subqueries) == 1:
            weight = self._weight_for(subqueries[0])
            return {subqueries[0].id: min(budget, max(1, weight))}

        raw_weights = [self._weight_for(subquery) for subquery in subqueries]
        total_weight = sum(raw_weights)
        allocations: dict[str, int] = {}
        remaining = budget

        for index, subquery in enumerate(subqueries):
            if index == len(subqueries) - 1:
                allocations[subquery.id] = max(1, remaining)
                continue

            share = max(1, round(budget * raw_weights[index] / total_weight))
            share = min(share, remaining - (len(subqueries) - index - 1))
            allocations[subquery.id] = share
            remaining -= share

        return allocations

    def _weight_for(self, subquery: SubQuery) -> int:
        return self._TYPE_WEIGHTS.get(subquery.query_type, 5)
