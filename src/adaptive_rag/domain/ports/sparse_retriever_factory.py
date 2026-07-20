"""Sparse retriever factory port."""

from __future__ import annotations

from typing import Protocol

from adaptive_rag.domain.models.index import IndexMetadata
from adaptive_rag.domain.ports.sparse_retriever import SparseRetrieverPort


class SparseRetrieverFactoryPort(Protocol):
    """Create sparse retriever adapters for a collection."""

    def create(self, *, collection_id: str, metadata: IndexMetadata) -> SparseRetrieverPort:
        """Instantiate a sparse retriever for the given collection."""
        ...

    def load(self, *, collection_id: str, path: str, metadata: IndexMetadata) -> SparseRetrieverPort:
        """Load a persisted sparse retriever for the given collection."""
        ...
