"""Index and vector storage domain models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from adaptive_rag.domain.models.document import Chunk
from adaptive_rag.domain.models.retrieval import SearchScope


class VectorBackend(str, Enum):
    """Supported dense vector storage backends."""

    FAISS = "faiss"
    QDRANT = "qdrant"
    MILVUS = "milvus"


class SparseBackend(str, Enum):
    """Supported sparse retrieval backends."""

    BM25 = "bm25"
    ELASTICSEARCH = "elasticsearch"


class FilterStrategy(str, Enum):
    """How metadata scope filters are applied during retrieval."""

    NATIVE = "native"
    POST_FILTER = "post_filter"


@dataclass(frozen=True)
class VectorStoreCapabilities:
    """Capability declaration for a dense vector store adapter."""

    version: str = "1.0"
    supports_native_filtering: bool = False
    supports_delete_by_document: bool = False
    supports_payload_updates: bool = False
    supports_hybrid_search: bool = False
    filter_strategy: FilterStrategy = FilterStrategy.POST_FILTER


@dataclass(frozen=True)
class SparseRetrieverCapabilities:
    """Capability declaration for a sparse retriever adapter."""

    version: str = "1.0"
    supports_native_filtering: bool = False
    supports_delete_by_document: bool = False
    filter_strategy: FilterStrategy = FilterStrategy.POST_FILTER


class IndexMetadata(BaseModel):
    """Metadata describing a collection index."""

    collection_id: str
    embedder_model: str
    embedder_dimension: int
    vector_backend: VectorBackend
    sparse_backend: SparseBackend
    schema_version: str = "1.0"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class VectorRecord(BaseModel):
    """Domain model for upserting a vector with minimal payload."""

    chunk_id: str
    document_id: str
    embedding: list[float]
    payload: dict[str, Any] = Field(default_factory=dict)
    chunk: Chunk | None = None


class VectorQuery(BaseModel):
    """Domain model for dense vector search."""

    embedding: list[float]
    scope: SearchScope
    top_k: int = 10


class CollectionStats(BaseModel):
    """Summary statistics for a collection index."""

    collection_id: str
    chunk_count: int
    dense_count: int
    sparse_count: int
