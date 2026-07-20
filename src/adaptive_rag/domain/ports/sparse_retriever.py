"""Sparse retriever port."""

from __future__ import annotations

from typing import Protocol

from adaptive_rag.domain.models.document import Chunk
from adaptive_rag.domain.models.index import SparseRetrieverCapabilities
from adaptive_rag.domain.models.retrieval import ScoredChunk, SearchScope


class SparseRetrieverPort(Protocol):
    """Sparse lexical retrieval."""

    @property
    def capabilities(self) -> SparseRetrieverCapabilities:
        """Return adapter capability declaration."""
        ...

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Build or update the sparse index."""
        ...

    def search(self, query: str, scope: SearchScope, top_k: int) -> list[ScoredChunk]:
        """Retrieve top-k chunks by sparse score within scope."""
        ...

    def delete_by_document_id(self, document_id: str) -> int:
        """Remove all chunks belonging to a document; returns deleted count."""
        ...

    def count(self) -> int:
        """Return number of indexed chunks."""
        ...

    def health_check(self) -> bool:
        """Return True when the retriever is operational."""
        ...

    def persist(self, path: str) -> None:
        """Persist the sparse index to disk (local backends only)."""
        ...

    def load(self, path: str) -> None:
        """Load a persisted sparse index from disk (local backends only)."""
        ...
