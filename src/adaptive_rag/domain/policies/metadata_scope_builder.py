"""Metadata scope construction from query analysis."""

from __future__ import annotations

from typing import Any

from adaptive_rag.domain.models.query import QueryAnalysis
from adaptive_rag.domain.models.retrieval import SearchScope
from adaptive_rag.domain.policies.metadata_filter import build_search_scope


class MetadataScopeBuilder:
    """Build SearchScope automatically from query analysis."""

    def build(
        self,
        *,
        analysis: QueryAnalysis,
        collection_id: str,
        estimated_corpus_size: int = 0,
        extra_filters: dict[str, Any] | None = None,
    ) -> SearchScope:
        """Derive scoped filters from analysis metadata_scope and optional overrides."""
        filters: dict[str, Any] = dict(analysis.metadata_scope)
        if extra_filters:
            filters.update(extra_filters)

        return build_search_scope(
            collection_id=collection_id,
            metadata_filters=filters or None,
            estimated_corpus_size=estimated_corpus_size,
        )
