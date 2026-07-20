"""Vector store port."""

from __future__ import annotations

from typing import Protocol

from adaptive_rag.domain.models.document import Chunk
from adaptive_rag.domain.models.index import VectorQuery, VectorRecord, VectorStoreCapabilities
from adaptive_rag.domain.models.retrieval import ScoredChunk, SearchScope


class VectorStorePort(Protocol):
    """Dense vector retrieval and indexing."""

    @property
    def capabilities(self) -> VectorStoreCapabilities:
        """Return adapter capability declaration."""
        ...

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Index chunks with precomputed embeddings."""
        ...

    def upsert(self, records: list[VectorRecord]) -> int:
        """Upsert vector records; returns count of records written."""
        ...

    def search(self, query: VectorQuery) -> list[ScoredChunk]:
        """Retrieve top-k chunks by dense similarity within scope."""
        ...

    def search_by_embedding(
        self,
        query_embedding: list[float],
        scope: SearchScope,
        top_k: int,
    ) -> list[ScoredChunk]:
        """Convenience search using raw embedding (delegates to VectorQuery)."""
        ...

    def delete_by_document_id(self, document_id: str) -> int:
        """Remove all chunks belonging to a document; returns deleted count."""
        ...

    def count(self) -> int:
        """Return number of indexed vectors."""
        ...

    def health_check(self) -> bool:
        """Return True when the store is operational."""
        ...

    def persist(self, path: str) -> None:
        """Persist the index to disk (local backends only)."""
        ...

    def load(self, path: str) -> None:
        """Load a persisted index from disk (local backends only)."""
        ...
