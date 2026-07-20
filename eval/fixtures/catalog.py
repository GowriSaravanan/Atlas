"""Build a chunk catalog for gold-label resolution."""

from __future__ import annotations

from typing import Any

from adaptive_rag.domain.models.retrieval import SearchScope
from adaptive_rag.domain.ports.index_registry import IndexRegistryPort

PROBE_QUERIES = [
    "HR-203",
    "HR-105",
    "annual leave",
    "sick leave",
    "maternity leave",
    "paternity leave",
    "handbook",
    "HR policy",
]


def build_chunk_catalog(
    registry: IndexRegistryPort,
    collection_id: str,
) -> list[tuple[str, str, dict[str, Any]]]:
    """Collect all chunks from a collection using broad probe queries."""
    stats = registry.stats(collection_id)
    scope = SearchScope(
        filters={"collection_id": collection_id},
        estimated_corpus_size=stats.chunk_count,
    )
    catalog: dict[str, tuple[str, str, dict[str, Any]]] = {}

    for probe in PROBE_QUERIES:
        hits = registry.search_sparse(collection_id, probe, scope, top_k=stats.chunk_count or 20)
        for hit in hits:
            catalog[hit.chunk.id] = (hit.chunk.id, hit.chunk.content, dict(hit.chunk.metadata))

    if not catalog:
        dense_hits = registry.search_dense(collection_id, "HR leave policy", scope, top_k=stats.chunk_count or 20)
        for hit in dense_hits:
            catalog[hit.chunk.id] = (hit.chunk.id, hit.chunk.content, dict(hit.chunk.metadata))

    return list(catalog.values())
