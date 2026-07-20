"""BM25 sparse retriever factory."""

from __future__ import annotations

from adaptive_rag.domain.models.index import IndexMetadata, SparseBackend
from adaptive_rag.domain.ports.sparse_retriever import SparseRetrieverPort
from adaptive_rag.domain.ports.sparse_retriever_factory import SparseRetrieverFactoryPort
from adaptive_rag.infrastructure.sparse.bm25_retriever import BM25SparseRetriever


class BM25SparseRetrieverFactory(SparseRetrieverFactoryPort):
    """Factory for BM25-backed sparse retrievers."""

    def create(self, *, collection_id: str, metadata: IndexMetadata) -> SparseRetrieverPort:
        if metadata.sparse_backend != SparseBackend.BM25:
            raise ValueError(f"BM25SparseRetrieverFactory cannot create {metadata.sparse_backend}")
        return BM25SparseRetriever(collection_id=collection_id)

    def load(self, *, collection_id: str, path: str, metadata: IndexMetadata) -> SparseRetrieverPort:
        retriever = self.create(collection_id=collection_id, metadata=metadata)
        retriever.load(path)
        return retriever
