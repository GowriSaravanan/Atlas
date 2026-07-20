"""Vector store factory port."""

from __future__ import annotations

from typing import Protocol

from adaptive_rag.domain.models.index import IndexMetadata
from adaptive_rag.domain.ports.vector_store import VectorStorePort


class VectorStoreFactoryPort(Protocol):
    """Create vector store adapters for a collection."""

    def create(self, *, collection_id: str, metadata: IndexMetadata) -> VectorStorePort:
        """Instantiate a vector store for the given collection."""
        ...

    def load(self, *, collection_id: str, path: str, metadata: IndexMetadata) -> VectorStorePort:
        """Load a persisted vector store for the given collection."""
        ...
