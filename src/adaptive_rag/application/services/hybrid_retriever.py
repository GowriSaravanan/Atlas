"""Hybrid retrieval service."""

from __future__ import annotations

import time
import uuid

from adaptive_rag.application.dto.retrieval import HybridRetrievalResult
from adaptive_rag.config.settings import RetrievalSettings
from adaptive_rag.domain.models.retrieval import RetrievalStrategy, ScoredChunk
from adaptive_rag.domain.models.trace import RetrievalTrace, StepTrace
from adaptive_rag.domain.policies.metadata_filter import build_search_scope
from adaptive_rag.domain.policies.retrieval_overlap import compute_retrieval_overlap
from adaptive_rag.domain.ports.fusion_engine import FusionEnginePort
from adaptive_rag.domain.ports.index_registry import IndexRegistryPort
from adaptive_rag.observability.logging import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    """Orchestrate BM25, dense, and RRF hybrid retrieval."""

    def __init__(
        self,
        *,
        index_registry: IndexRegistryPort,
        fusion_engine: FusionEnginePort,
        settings: RetrievalSettings,
    ) -> None:
        self._index_registry = index_registry
        self._fusion_engine = fusion_engine
        self._settings = settings

    def retrieve(
        self,
        *,
        query: str,
        collection_id: str,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
        metadata_filters: dict | None = None,
        top_k: int | None = None,
    ) -> HybridRetrievalResult:
        """Run retrieval using the requested strategy."""
        top_k = top_k or self._settings.default_top_k
        query_id = str(uuid.uuid4())
        trace = RetrievalTrace(query_id=query_id, raw_query=query, strategy=strategy)
        latency: dict[str, float] = {}

        stats = self._index_registry.stats(collection_id)
        scope = build_search_scope(
            collection_id=collection_id,
            metadata_filters=metadata_filters,
            estimated_corpus_size=stats.chunk_count,
        )
        trace.search_scope = scope

        dense_hits: list[ScoredChunk] = []
        sparse_hits: list[ScoredChunk] = []
        fused_hits: list[ScoredChunk] = []
        results: list[ScoredChunk] = []

        if strategy in (RetrievalStrategy.DENSE, RetrievalStrategy.HYBRID):
            start = time.perf_counter()
            dense_hits = self._index_registry.search_dense(collection_id, query, scope, top_k)
            latency["dense_ms"] = (time.perf_counter() - start) * 1000
            trace.steps.append(
                StepTrace(step="dense_retrieval", duration_ms=latency["dense_ms"], metadata={"hits": len(dense_hits)})
            )

        if strategy in (RetrievalStrategy.BM25, RetrievalStrategy.HYBRID):
            start = time.perf_counter()
            sparse_hits = self._index_registry.search_sparse(collection_id, query, scope, top_k)
            latency["sparse_ms"] = (time.perf_counter() - start) * 1000
            trace.steps.append(
                StepTrace(step="sparse_retrieval", duration_ms=latency["sparse_ms"], metadata={"hits": len(sparse_hits)})
            )

        if strategy == RetrievalStrategy.HYBRID:
            start = time.perf_counter()
            ranked_lists: dict[str, list[ScoredChunk]] = {}
            if dense_hits:
                ranked_lists["dense"] = dense_hits
            if sparse_hits:
                ranked_lists["bm25"] = sparse_hits

            fused_hits = self._fusion_engine.fuse(ranked_lists, query)[:top_k]
            latency["fusion_ms"] = (time.perf_counter() - start) * 1000
            trace.steps.append(
                StepTrace(step="rrf_fusion", duration_ms=latency["fusion_ms"], metadata={"hits": len(fused_hits)})
            )
            results = fused_hits
        elif strategy == RetrievalStrategy.DENSE:
            results = dense_hits
        else:
            results = sparse_hits

        overlap = compute_retrieval_overlap(dense_hits, sparse_hits, top_k=top_k)
        trace.dense_hits = dense_hits
        trace.sparse_hits = sparse_hits
        trace.fused_hits = fused_hits
        trace.latency_ms = latency

        logger.info(
            "Hybrid retrieval completed",
            extra={
                "ctx_query_id": query_id,
                "ctx_strategy": strategy.value,
                "ctx_results": len(results),
                "ctx_overlap": overlap,
            },
        )

        return HybridRetrievalResult(
            query=query,
            collection_id=collection_id,
            strategy=strategy,
            results=results,
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
            fused_hits=fused_hits,
            retrieval_overlap=overlap,
            trace=trace,
        )
