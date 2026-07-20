"""Index registry port."""

from __future__ import annotations

from typing import Protocol

from adaptive_rag.domain.models.document import Chunk
from adaptive_rag.domain.models.index import CollectionStats, IndexMetadata
from adaptive_rag.domain.models.retrieval import ScoredChunk, SearchScope


class IndexRegistryPort(Protocol):
    """Application-facing port for document indexing and collection management."""

    def index_chunks(self, collection_id: str, chunks: list[Chunk]) -> None:
        """Embed and index chunks into dense and sparse stores."""
        ...

    def get_index_metadata(self, collection_id: str) -> IndexMetadata:
        """Return metadata for a collection index."""
        ...

    def stats(self, collection_id: str) -> CollectionStats:
        """Return indexing statistics for a collection."""
        ...

    def persist(self, collection_id: str) -> None:
        """Persist collection indexes to storage."""
        ...

    def health_check(self, collection_id: str) -> bool:
        """Return True when both dense and sparse indexes are healthy."""
        ...

    def list_collections(self) -> list[str]:
        """List known collection identifiers."""
        ...

    def search_dense(
        self,
        collection_id: str,
        query: str,
        scope: SearchScope,
        top_k: int,
    ) -> list[ScoredChunk]:
        """Run dense retrieval for a collection."""
        ...

    def search_sparse(
        self,
        collection_id: str,
        query: str,
        scope: SearchScope,
        top_k: int,
    ) -> list[ScoredChunk]:
        """Run sparse retrieval for a collection."""
        ...
