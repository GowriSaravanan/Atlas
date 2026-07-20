"""Metadata scope construction for retrieval."""

from __future__ import annotations

from typing import Any

from adaptive_rag.domain.models.retrieval import SearchScope


def build_search_scope(
    *,
    collection_id: str,
    metadata_filters: dict[str, Any] | None = None,
    estimated_corpus_size: int = 0,
) -> SearchScope:
    """Build a search scope from collection id and optional metadata constraints."""
    filters: dict[str, Any] = {"collection_id": collection_id}
    if metadata_filters:
        filters.update(metadata_filters)
    return SearchScope(
        filters=filters,
        estimated_corpus_size=estimated_corpus_size,
        scope_metadata={"collection_id": collection_id},
    )
