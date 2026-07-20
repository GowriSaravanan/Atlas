"""Sparse retriever port."""

from __future__ import annotations

from typing import Protocol

from adaptive_rag.domain.models.document import Chunk
from adaptive_rag.domain.models.retrieval import ScoredChunk, SearchScope


class SparseRetrieverPort(Protocol):
    """BM25 sparse retrieval."""

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Build or update the sparse index."""
        ...

    def search(self, query: str, scope: SearchScope, top_k: int) -> list[ScoredChunk]:
        """Retrieve top-k chunks by BM25 score within scope."""
        ...

    def persist(self, path: str) -> None:
        """Persist the sparse index to disk."""
        ...

    def load(self, path: str) -> None:
        """Load a persisted sparse index from disk."""
        ...
