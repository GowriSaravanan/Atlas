"""FAISS vector store factory."""

from __future__ import annotations

from adaptive_rag.domain.models.index import IndexMetadata, VectorBackend
from adaptive_rag.domain.ports.vector_store import VectorStorePort
from adaptive_rag.domain.ports.vector_store_factory import VectorStoreFactoryPort
from adaptive_rag.infrastructure.vector_store.faiss_store import FaissVectorStore


class FaissVectorStoreFactory(VectorStoreFactoryPort):
    """Factory for FAISS-backed vector stores."""

    def create(self, *, collection_id: str, metadata: IndexMetadata) -> VectorStorePort:
        if metadata.vector_backend != VectorBackend.FAISS:
            raise ValueError(f"FaissVectorStoreFactory cannot create {metadata.vector_backend}")
        return FaissVectorStore(
            dimension=metadata.embedder_dimension,
            collection_id=collection_id,
        )

    def load(self, *, collection_id: str, path: str, metadata: IndexMetadata) -> VectorStorePort:
        store = self.create(collection_id=collection_id, metadata=metadata)
        store.load(path)
        return store
