"""Provider factory wiring for vector and sparse backends."""

from __future__ import annotations

from adaptive_rag.config.settings import Settings
from adaptive_rag.domain.models.index import VectorBackend
from adaptive_rag.domain.ports.sparse_retriever_factory import SparseRetrieverFactoryPort
from adaptive_rag.domain.ports.vector_store_factory import VectorStoreFactoryPort
from adaptive_rag.infrastructure.sparse.bm25_factory import BM25SparseRetrieverFactory
from adaptive_rag.infrastructure.vector_store.faiss_factory import FaissVectorStoreFactory


def build_vector_store_factory(settings: Settings) -> VectorStoreFactoryPort:
    """Return the configured vector store factory."""
    if settings.vector_store.provider == VectorBackend.FAISS:
        return FaissVectorStoreFactory()
    raise NotImplementedError(
        f"Vector store provider '{settings.vector_store.provider.value}' is not implemented yet",
    )


def build_sparse_retriever_factory(settings: Settings) -> SparseRetrieverFactoryPort:
    """Return the configured sparse retriever factory."""
    return BM25SparseRetrieverFactory()
