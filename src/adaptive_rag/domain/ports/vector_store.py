"""Vector store port."""

from __future__ import annotations

from typing import Protocol

from adaptive_rag.domain.models.document import Chunk
from adaptive_rag.domain.models.retrieval import ScoredChunk, SearchScope


class VectorStorePort(Protocol):
    """Dense vector retrieval and indexing."""

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Index chunks with precomputed embeddings."""
        ...

    def search(
        self,
        query_embedding: list[float],
        scope: SearchScope,
        top_k: int,
    ) -> list[ScoredChunk]:
        """Retrieve top-k chunks by dense similarity within scope."""
        ...

    def delete_by_document_id(self, document_id: str) -> None:
        """Remove all chunks belonging to a document."""
        ...

    def persist(self, path: str) -> None:
        """Persist the index to disk."""
        ...

    def load(self, path: str) -> None:
        """Load a persisted index from disk."""
        ...
