"""FAISS vector store adapter."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from adaptive_rag.domain.models.document import Chunk
from adaptive_rag.domain.models.index import (
    FilterStrategy,
    VectorQuery,
    VectorRecord,
    VectorStoreCapabilities,
)
from adaptive_rag.domain.models.retrieval import ScoredChunk, SearchScope
from adaptive_rag.domain.policies.scope_filter import chunk_matches_scope
from adaptive_rag.domain.ports.vector_store import VectorStorePort
from adaptive_rag.observability.logging import get_logger

logger = get_logger(__name__)

FAISS_CAPABILITIES = VectorStoreCapabilities(
    version="1.0",
    supports_native_filtering=False,
    supports_delete_by_document=True,
    supports_payload_updates=False,
    supports_hybrid_search=False,
    filter_strategy=FilterStrategy.POST_FILTER,
)


class FaissVectorStore(VectorStorePort):
    """In-memory FAISS index with JSON sidecar for chunk metadata."""

    _INDEX_FILE = "faiss.index"
    _CHUNKS_FILE = "chunks.json"

    def __init__(self, *, dimension: int, collection_id: str) -> None:
        self._collection_id = collection_id
        self._dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self._chunks: list[Chunk] = []
        self._chunk_ids: set[str] = set()

    @property
    def capabilities(self) -> VectorStoreCapabilities:
        return FAISS_CAPABILITIES

    @property
    def collection_id(self) -> str:
        return self._collection_id

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        records = [
            VectorRecord(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                embedding=embedding,
                payload=dict(chunk.metadata),
                chunk=chunk,
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        self.upsert(records)

    def upsert(self, records: list[VectorRecord]) -> int:
        if not records:
            return 0

        new_chunks: list[Chunk] = []
        new_vectors: list[list[float]] = []
        for record in records:
            if record.chunk_id in self._chunk_ids:
                continue
            chunk = record.chunk or Chunk(
                id=record.chunk_id,
                document_id=record.document_id,
                content=str(record.payload.get("content", "")),
                metadata=record.payload,
            )
            scoped_chunk = self._ensure_collection_metadata(chunk)
            new_chunks.append(scoped_chunk)
            new_vectors.append(record.embedding)

        if not new_chunks:
            return 0

        matrix = np.array(new_vectors, dtype=np.float32)
        faiss.normalize_L2(matrix)
        self._index.add(matrix)
        self._chunks.extend(new_chunks)
        self._chunk_ids.update(chunk.id for chunk in new_chunks)

        logger.info(
            "Added chunks to FAISS index",
            extra={"ctx_collection_id": self._collection_id, "ctx_added": len(new_chunks)},
        )
        return len(new_chunks)

    def search(self, query: VectorQuery) -> list[ScoredChunk]:
        return self.search_by_embedding(query.embedding, query.scope, query.top_k)

    def search_by_embedding(
        self,
        query_embedding: list[float],
        scope: SearchScope,
        top_k: int,
    ) -> list[ScoredChunk]:
        if self._index.ntotal == 0:
            return []

        query = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query)

        fetch_k = min(max(top_k * 5, top_k), self._index.ntotal)
        scores, indices = self._index.search(query, fetch_k)

        results: list[ScoredChunk] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0], strict=True)):
            if idx < 0 or idx >= len(self._chunks):
                continue
            chunk = self._chunks[int(idx)]
            if not chunk_matches_scope(chunk, scope):
                continue
            results.append(
                ScoredChunk(
                    chunk=chunk,
                    score=float(score),
                    source="dense",
                    rank=rank + 1,
                )
            )
            if len(results) >= top_k:
                break

        return results

    def delete_by_document_id(self, document_id: str) -> int:
        remaining = [chunk for chunk in self._chunks if chunk.document_id != document_id]
        deleted = len(self._chunks) - len(remaining)
        if deleted == 0:
            return 0
        self._rebuild_index(remaining)
        return deleted

    def count(self) -> int:
        return len(self._chunks)

    def health_check(self) -> bool:
        return self._index is not None and self._dimension > 0

    def persist(self, path: str) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(target / self._INDEX_FILE))
        payload = {
            "collection_id": self._collection_id,
            "dimension": self._dimension,
            "chunks": [chunk.model_dump(mode="json") for chunk in self._chunks],
        }
        (target / self._CHUNKS_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, path: str) -> None:
        target = Path(path)
        index_path = target / self._INDEX_FILE
        chunks_path = target / self._CHUNKS_FILE
        if not index_path.exists() or not chunks_path.exists():
            return

        self._index = faiss.read_index(str(index_path))
        payload = json.loads(chunks_path.read_text(encoding="utf-8"))
        self._collection_id = payload.get("collection_id", self._collection_id)
        self._dimension = int(payload.get("dimension", self._dimension))
        self._chunks = [Chunk.model_validate(item) for item in payload.get("chunks", [])]
        self._chunk_ids = {chunk.id for chunk in self._chunks}

    def _rebuild_index(self, chunks: list[Chunk]) -> None:
        self._index = faiss.IndexFlatIP(self._dimension)
        self._chunks = []
        self._chunk_ids = set()
        if not chunks:
            return
        self._chunks = chunks
        self._chunk_ids = {chunk.id for chunk in chunks}

    def _ensure_collection_metadata(self, chunk: Chunk) -> Chunk:
        metadata = dict(chunk.metadata)
        metadata["collection_id"] = self._collection_id
        return chunk.model_copy(update={"metadata": metadata})
