"""BM25 sparse retriever adapter."""

from __future__ import annotations

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from adaptive_rag.domain.models.document import Chunk
from adaptive_rag.domain.models.index import FilterStrategy, SparseRetrieverCapabilities
from adaptive_rag.domain.models.retrieval import ScoredChunk, SearchScope
from adaptive_rag.domain.policies.scope_filter import chunk_matches_scope
from adaptive_rag.domain.ports.sparse_retriever import SparseRetrieverPort
from adaptive_rag.observability.logging import get_logger

logger = get_logger(__name__)

BM25_CAPABILITIES = SparseRetrieverCapabilities(
    version="1.0",
    supports_native_filtering=False,
    supports_delete_by_document=True,
    filter_strategy=FilterStrategy.POST_FILTER,
)


class BM25SparseRetriever(SparseRetrieverPort):
    """BM25 sparse index with JSON sidecar persistence."""

    _CORPUS_FILE = "bm25_corpus.json"

    def __init__(self, *, collection_id: str) -> None:
        self._collection_id = collection_id
        self._chunks: list[Chunk] = []
        self._chunk_ids: set[str] = set()
        self._bm25: BM25Okapi | None = None
        self._tokenized_corpus: list[list[str]] = []

    @property
    def capabilities(self) -> SparseRetrieverCapabilities:
        return BM25_CAPABILITIES

    @property
    def collection_id(self) -> str:
        return self._collection_id

    def index_chunks(self, chunks: list[Chunk]) -> None:
        new_chunks: list[Chunk] = []
        for chunk in chunks:
            if chunk.id in self._chunk_ids:
                continue
            metadata = dict(chunk.metadata)
            metadata["collection_id"] = self._collection_id
            new_chunks.append(chunk.model_copy(update={"metadata": metadata}))

        if not new_chunks:
            return

        self._chunks.extend(new_chunks)
        self._chunk_ids.update(chunk.id for chunk in new_chunks)
        self._rebuild_bm25()

        logger.info(
            "Indexed chunks in BM25 corpus",
            extra={"ctx_collection_id": self._collection_id, "ctx_added": len(new_chunks)},
        )

    def search(self, query: str, scope: SearchScope, top_k: int) -> list[ScoredChunk]:
        if self._bm25 is None or not self._chunks:
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)

        results: list[ScoredChunk] = []
        for rank, idx in enumerate(ranked_indices, start=1):
            chunk = self._chunks[idx]
            if not chunk_matches_scope(chunk, scope):
                continue
            score = float(scores[idx])
            results.append(
                ScoredChunk(
                    chunk=chunk,
                    score=score,
                    source="bm25",
                    rank=rank,
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
        self._chunks = remaining
        self._chunk_ids = {chunk.id for chunk in remaining}
        self._rebuild_bm25()
        return deleted

    def count(self) -> int:
        return len(self._chunks)

    def health_check(self) -> bool:
        return self._bm25 is not None or len(self._chunks) == 0

    def persist(self, path: str) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        payload = {
            "collection_id": self._collection_id,
            "chunks": [chunk.model_dump(mode="json") for chunk in self._chunks],
        }
        (target / self._CORPUS_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, path: str) -> None:
        target = Path(path)
        corpus_path = target / self._CORPUS_FILE
        if not corpus_path.exists():
            return

        payload = json.loads(corpus_path.read_text(encoding="utf-8"))
        self._collection_id = payload.get("collection_id", self._collection_id)
        self._chunks = [Chunk.model_validate(item) for item in payload.get("chunks", [])]
        self._chunk_ids = {chunk.id for chunk in self._chunks}
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        self._tokenized_corpus = [self._tokenize(chunk.content) for chunk in self._chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus) if self._tokenized_corpus else None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9]+", text.lower())
