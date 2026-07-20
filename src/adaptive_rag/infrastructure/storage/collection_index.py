"""Collection-level index management."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from adaptive_rag.domain.errors import EmbedderCompatibilityError
from adaptive_rag.domain.models.document import Chunk
from adaptive_rag.domain.models.index import (
    CollectionStats,
    IndexMetadata,
    SparseBackend,
    VectorBackend,
    VectorQuery,
)
from adaptive_rag.domain.models.retrieval import ScoredChunk, SearchScope
from adaptive_rag.domain.ports.embedder import EmbedderPort
from adaptive_rag.domain.ports.index_registry import IndexRegistryPort
from adaptive_rag.domain.ports.sparse_retriever import SparseRetrieverPort
from adaptive_rag.domain.ports.sparse_retriever_factory import SparseRetrieverFactoryPort
from adaptive_rag.domain.ports.vector_store import VectorStorePort
from adaptive_rag.domain.ports.vector_store_factory import VectorStoreFactoryPort
from adaptive_rag.domain.validation.collection_id import resolve_collection_path, validate_collection_id
from adaptive_rag.observability.logging import get_logger

logger = get_logger(__name__)

_METADATA_FILE = "index_metadata.json"


class CollectionIndex:
    """Manages dense and sparse indexes for a single collection."""

    def __init__(
        self,
        *,
        metadata: IndexMetadata,
        vector_store: VectorStorePort,
        sparse_retriever: SparseRetrieverPort,
        embedder: EmbedderPort,
    ) -> None:
        self.metadata = metadata
        self._vector_store = vector_store
        self._sparse_retriever = sparse_retriever
        self._embedder = embedder

    @property
    def collection_id(self) -> str:
        return self.metadata.collection_id

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Embed and index chunks in both dense and sparse stores."""
        if not chunks:
            return

        embeddings = self._embedder.embed_chunks(chunks)
        self._vector_store.add_chunks(chunks, embeddings)
        self._sparse_retriever.index_chunks(chunks)
        self.metadata = self.metadata.model_copy(
            update={"updated_at": datetime.now(UTC)},
        )

    def search_dense(self, query: str, scope: SearchScope, top_k: int) -> list[ScoredChunk]:
        """Run dense retrieval against this collection."""
        embedding = self._embedder.embed_query(query)
        vector_query = VectorQuery(embedding=embedding, scope=scope, top_k=top_k)
        return self._vector_store.search(vector_query)

    def search_sparse(self, query: str, scope: SearchScope, top_k: int) -> list[ScoredChunk]:
        """Run BM25 retrieval against this collection."""
        return self._sparse_retriever.search(query, scope, top_k)

    def stats(self) -> CollectionStats:
        dense_count = self._vector_store.count()
        sparse_count = self._sparse_retriever.count()
        return CollectionStats(
            collection_id=self.collection_id,
            chunk_count=max(dense_count, sparse_count),
            dense_count=dense_count,
            sparse_count=sparse_count,
        )

    def health_check(self) -> bool:
        return self._vector_store.health_check() and self._sparse_retriever.health_check()

    def persist(self, base_path: Path) -> None:
        collection_path = base_path / self.collection_id
        collection_path.mkdir(parents=True, exist_ok=True)
        self._vector_store.persist(str(collection_path / "dense"))
        self._sparse_retriever.persist(str(collection_path / "sparse"))
        (collection_path / _METADATA_FILE).write_text(
            self.metadata.model_dump_json(indent=2),
            encoding="utf-8",
        )
        logger.info(
            "Persisted collection index",
            extra={"ctx_collection_id": self.collection_id, "ctx_path": str(collection_path)},
        )


class CollectionIndexRegistry(IndexRegistryPort):
    """Registry for collection indexes with disk persistence."""

    def __init__(
        self,
        *,
        base_path: Path,
        embedder: EmbedderPort,
        vector_store_factory: VectorStoreFactoryPort,
        sparse_retriever_factory: SparseRetrieverFactoryPort,
        vector_backend: VectorBackend,
        sparse_backend: SparseBackend,
    ) -> None:
        self._base_path = base_path
        self._embedder = embedder
        self._vector_store_factory = vector_store_factory
        self._sparse_retriever_factory = sparse_retriever_factory
        self._vector_backend = vector_backend
        self._sparse_backend = sparse_backend
        self._collections: dict[str, CollectionIndex] = {}
        self._base_path.mkdir(parents=True, exist_ok=True)

    @property
    def embedder(self) -> EmbedderPort:
        return self._embedder

    def index_chunks(self, collection_id: str, chunks: list[Chunk]) -> None:
        validate_collection_id(collection_id)
        index = self.get_or_create(collection_id)
        index.add_chunks(chunks)
        self.persist(collection_id)

    def get_index_metadata(self, collection_id: str) -> IndexMetadata:
        return self.get_or_create(collection_id).metadata

    def stats(self, collection_id: str) -> CollectionStats:
        return self.get_or_create(collection_id).stats()

    def persist(self, collection_id: str) -> None:
        index = self.get_or_create(collection_id)
        index.persist(self._base_path)

    def health_check(self, collection_id: str) -> bool:
        return self.get_or_create(collection_id).health_check()

    def list_collections(self) -> list[str]:
        collections = {path.name for path in self._base_path.iterdir() if path.is_dir()}
        collections.update(self._collections.keys())
        return sorted(collections)

    def search_dense(
        self,
        collection_id: str,
        query: str,
        scope: SearchScope,
        top_k: int,
    ) -> list[ScoredChunk]:
        return self.get_or_create(collection_id).search_dense(query, scope, top_k)

    def search_sparse(
        self,
        collection_id: str,
        query: str,
        scope: SearchScope,
        top_k: int,
    ) -> list[ScoredChunk]:
        return self.get_or_create(collection_id).search_sparse(query, scope, top_k)

    def get_or_create(self, collection_id: str) -> CollectionIndex:
        validate_collection_id(collection_id)
        if collection_id in self._collections:
            return self._collections[collection_id]

        collection_path = resolve_collection_path(self._base_path, collection_id)
        metadata = self._load_or_create_metadata(collection_id, collection_path)

        if collection_path.exists() and (collection_path / _METADATA_FILE).exists():
            index = self._load_index(collection_id, collection_path, metadata)
        else:
            index = self._create_index(metadata)

        self._collections[collection_id] = index
        return index

    def _load_or_create_metadata(self, collection_id: str, collection_path: Path) -> IndexMetadata:
        metadata_path = collection_path / _METADATA_FILE
        if metadata_path.exists():
            return IndexMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))

        now = datetime.now(UTC)
        return IndexMetadata(
            collection_id=collection_id,
            embedder_model=self._embedder.model_name,
            embedder_dimension=self._embedder.dimension,
            vector_backend=self._vector_backend,
            sparse_backend=self._sparse_backend,
            created_at=now,
            updated_at=now,
        )

    def _create_index(self, metadata: IndexMetadata) -> CollectionIndex:
        vector_store = self._vector_store_factory.create(
            collection_id=metadata.collection_id,
            metadata=metadata,
        )
        sparse_retriever = self._sparse_retriever_factory.create(
            collection_id=metadata.collection_id,
            metadata=metadata,
        )
        return CollectionIndex(
            metadata=metadata,
            vector_store=vector_store,
            sparse_retriever=sparse_retriever,
            embedder=self._embedder,
        )

    def _load_index(
        self,
        collection_id: str,
        collection_path: Path,
        metadata: IndexMetadata,
    ) -> CollectionIndex:
        self._validate_embedder_compatibility(metadata)
        vector_store = self._vector_store_factory.load(
            collection_id=collection_id,
            path=str(collection_path / "dense"),
            metadata=metadata,
        )
        sparse_retriever = self._sparse_retriever_factory.load(
            collection_id=collection_id,
            path=str(collection_path / "sparse"),
            metadata=metadata,
        )
        return CollectionIndex(
            metadata=metadata,
            vector_store=vector_store,
            sparse_retriever=sparse_retriever,
            embedder=self._embedder,
        )

    def _validate_embedder_compatibility(self, metadata: IndexMetadata) -> None:
        if metadata.embedder_model != self._embedder.model_name:
            logger.warning(
                "Embedder model mismatch for collection",
                extra={
                    "ctx_collection_id": metadata.collection_id,
                    "ctx_index_model": metadata.embedder_model,
                    "ctx_current_model": self._embedder.model_name,
                },
            )
        if metadata.embedder_dimension != self._embedder.dimension:
            raise EmbedderCompatibilityError(
                f"Embedder dimension mismatch for collection {metadata.collection_id}: "
                f"index={metadata.embedder_dimension}, current={self._embedder.dimension}",
            )
