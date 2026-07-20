"""Top-level adaptive retrieval orchestration."""

from __future__ import annotations

import time
import uuid

from adaptive_rag.application.dto.retrieval import RetrievalEngineResult
from adaptive_rag.application.services.hybrid_retriever import HybridRetriever
from adaptive_rag.config.settings import RetrievalSettings
from adaptive_rag.domain.models.confidence import ConfidenceScore
from adaptive_rag.domain.models.conversation import Message
from adaptive_rag.domain.models.decomposition import (
    DecompositionResult,
    SubQuery,
    SubQueryResult,
    SubQueryRetrievalPlan,
)
from adaptive_rag.domain.models.query import (
    OriginalQueryAnalysis,
    ResolvedQueryAnalysis,
    RetrievalDecision,
    RewriteResult,
)
from adaptive_rag.domain.models.retrieval import RetrievalStrategy, ScoredChunk
from adaptive_rag.domain.models.trace import RetrievalTrace, StepTrace
from adaptive_rag.domain.policies.adaptive_router import AdaptiveRouter
from adaptive_rag.domain.policies.confidence import ConfidenceScorer
from adaptive_rag.domain.policies.metadata_scope_builder import MetadataScopeBuilder
from adaptive_rag.domain.policies.query_analyzer import QueryAnalyzer
from adaptive_rag.domain.policies.query_decomposer import RuleBasedQueryDecomposer
from adaptive_rag.domain.policies.query_rewriter import RuleBasedQueryRewriter
from adaptive_rag.domain.policies.subquery_result_merger import SubqueryResultMerger
from adaptive_rag.domain.policies.top_k_allocator import TopKAllocator
from adaptive_rag.domain.ports.fusion_engine import FusionEnginePort
from adaptive_rag.domain.ports.index_registry import IndexRegistryPort
from adaptive_rag.domain.ports.query_decomposer import QueryDecomposerPort
from adaptive_rag.domain.ports.query_rewriter import QueryRewriterPort
from adaptive_rag.domain.ports.reranker import RerankerPort
from adaptive_rag.domain.ports.answer_generator import AnswerGeneratorPort
from adaptive_rag.observability.logging import get_logger

logger = get_logger(__name__)


class RetrievalEngine:
    """Orchestrate query analysis, rewrite, decomposition, retrieval, and confidence."""

    def __init__(
        self,
        *,
        index_registry: IndexRegistryPort,
        hybrid_retriever: HybridRetriever,
        settings: RetrievalSettings,
        fusion_engine: FusionEnginePort,
        query_analyzer: QueryAnalyzer | None = None,
        query_rewriter: QueryRewriterPort | None = None,
        query_decomposer: QueryDecomposerPort | None = None,
        adaptive_router: AdaptiveRouter | None = None,
        scope_builder: MetadataScopeBuilder | None = None,
        confidence_scorer: ConfidenceScorer | None = None,
        top_k_allocator: TopKAllocator | None = None,
        result_merger: SubqueryResultMerger | None = None,
        reranker: RerankerPort | None = None,
        answer_generator: AnswerGeneratorPort | None = None,
    ) -> None:
        self._index_registry = index_registry
        self._hybrid_retriever = hybrid_retriever
        self._settings = settings
        self._query_analyzer = query_analyzer or QueryAnalyzer()
        self._query_rewriter = query_rewriter or RuleBasedQueryRewriter()
        self._query_decomposer = query_decomposer or RuleBasedQueryDecomposer(self._query_analyzer)
        self._adaptive_router = adaptive_router or AdaptiveRouter()
        self._scope_builder = scope_builder or MetadataScopeBuilder()
        self._confidence_scorer = confidence_scorer
        if self._confidence_scorer is None:
            raise ValueError("confidence_scorer must be provided")
        self._top_k_allocator = top_k_allocator or TopKAllocator()
        self._result_merger = result_merger or SubqueryResultMerger(fusion_engine)
        self._reranker = reranker
        self._answer_generator = answer_generator

    def execute(
        self,
        *,
        query: str,
        collection_id: str = "default",
        strategy: RetrievalStrategy | None = None,
        metadata_filters: dict | None = None,
        top_k: int | None = None,
        context_messages: list[Message] | None = None,
    ) -> RetrievalEngineResult:
        """Run the full adaptive retrieval pipeline."""
        query_id = str(uuid.uuid4())
        total_top_k = top_k or self._settings.default_top_k

        start = time.perf_counter()
        original_analysis: OriginalQueryAnalysis = self._query_analyzer.analyze(query)
        original_analyze_ms = (time.perf_counter() - start) * 1000

        rewrite_result = self._maybe_rewrite(
            query=query,
            analysis=original_analysis,
            context_messages=context_messages or [],
        )
        retrieval_query = rewrite_result.rewritten_query

        resolved_analysis: ResolvedQueryAnalysis
        resolved_analyze_ms = 0.0
        if rewrite_result.was_rewritten:
            start = time.perf_counter()
            resolved_analysis = self._query_analyzer.analyze(retrieval_query)
            resolved_analyze_ms = (time.perf_counter() - start) * 1000
        else:
            resolved_analysis = original_analysis

        decomposition = self._query_decomposer.decompose(retrieval_query, resolved_analysis)
        top_k_by_subquery = self._top_k_allocator.allocate(decomposition.subqueries, total_top_k)

        stats = self._index_registry.stats(collection_id)
        subquery_results: list[SubQueryResult] = []
        for subquery in decomposition.subqueries:
            subquery_results.append(
                self._retrieve_subquery(
                    subquery=subquery,
                    collection_id=collection_id,
                    metadata_filters=metadata_filters,
                    explicit_strategy=strategy,
                    top_k=top_k_by_subquery[subquery.id],
                    corpus_size=stats.chunk_count,
                )
            )

        merged_results = self._result_merger.merge(
            subquery_results,
            top_k=total_top_k,
            parent_query=retrieval_query,
        )
        pre_rerank_results = list(merged_results)
        rerank_ms = 0.0
        rerank_metadata: dict[str, object] = {
            "input_count": len(pre_rerank_results),
            "output_count": len(pre_rerank_results),
            "pre_rerank_ids": [hit.chunk.id for hit in pre_rerank_results],
            "post_rerank_ids": [hit.chunk.id for hit in pre_rerank_results],
            "skipped": self._reranker is None,
        }
        if self._reranker is not None and pre_rerank_results:
            start = time.perf_counter()
            merged_results = self._reranker.rerank(
                retrieval_query,
                pre_rerank_results,
                top_k=self._settings.rerank_top_k,
            )
            rerank_ms = (time.perf_counter() - start) * 1000
            rerank_metadata = {
                "input_count": len(pre_rerank_results),
                "output_count": len(merged_results),
                "pre_rerank_ids": [hit.chunk.id for hit in pre_rerank_results],
                "post_rerank_ids": [hit.chunk.id for hit in merged_results],
                "skipped": False,
                "top_k": self._settings.rerank_top_k,
            }

        decision = self._build_aggregate_decision(subquery_results, explicit_strategy=strategy)
        confidence = self._aggregate_confidence(subquery_results, resolved_analysis, merged_results)

        dense_hits = self._dedupe_hits(
            [hit for result in subquery_results for hit in result.dense_hits]
        )
        sparse_hits = self._dedupe_hits(
            [hit for result in subquery_results for hit in result.sparse_hits]
        )
        fused_hits = self._resolve_fused_hits(subquery_results, merged_results)
        overlap_values = [result.retrieval_overlap for result in subquery_results]
        retrieval_overlap = sum(overlap_values) / len(overlap_values) if overlap_values else 0.0

        trace = RetrievalTrace(
            query_id=query_id,
            raw_query=query,
            resolved_query=retrieval_query,
            original_analysis=original_analysis,
            resolved_analysis=resolved_analysis,
            strategy=decision.strategy,
            search_scope=self._scope_builder.build(
                analysis=resolved_analysis,
                collection_id=collection_id,
                estimated_corpus_size=stats.chunk_count,
                extra_filters=metadata_filters,
            ),
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
            fused_hits=fused_hits,
            reranked_hits=merged_results,
            retrieval_confidence=confidence,
            latency_ms={"original_analyze_ms": original_analyze_ms},
        )
        if rerank_ms > 0.0:
            trace.latency_ms["rerank_ms"] = rerank_ms
        if rewrite_result.was_rewritten:
            trace.latency_ms["resolved_analyze_ms"] = resolved_analyze_ms
        for subquery_result in subquery_results:
            for key, value in subquery_result.trace.latency_ms.items():
                trace.latency_ms[key] = trace.latency_ms.get(key, 0.0) + value

        self._append_trace_steps(
            trace=trace,
            original_analysis=original_analysis,
            resolved_analysis=resolved_analysis,
            rewrite_result=rewrite_result,
            retrieval_query=retrieval_query,
            original_analyze_ms=original_analyze_ms,
            resolved_analyze_ms=resolved_analyze_ms,
            decomposition=decomposition,
            subquery_results=subquery_results,
            decision=decision,
            confidence=confidence,
            merged_count=len(pre_rerank_results),
            rerank_ms=rerank_ms,
            rerank_metadata=rerank_metadata,
        )

        generated_answer = None
        answer_ms = 0.0
        if self._answer_generator is not None:
            start = time.perf_counter()
            generated_answer = self._answer_generator.generate(retrieval_query, merged_results)
            answer_ms = (time.perf_counter() - start) * 1000
            trace.steps.append(
                StepTrace(
                    step="answer_generation",
                    duration_ms=answer_ms,
                    metadata={
                        "model_name": generated_answer.model_name,
                        "used_chunk_ids": generated_answer.used_chunk_ids,
                        "prompt_tokens": generated_answer.prompt_tokens,
                        "completion_tokens": generated_answer.completion_tokens,
                        "answer_length": len(generated_answer.answer),
                    },
                )
            )
            trace.latency_ms["answer_generation_ms"] = answer_ms

        logger.info(
            "Retrieval engine completed",
            extra={
                "ctx_query_id": query_id,
                "ctx_strategy": decision.strategy.value,
                "ctx_confidence": confidence.value,
                "ctx_rewritten": rewrite_result.was_rewritten,
                "ctx_decomposed": decomposition.was_decomposed,
                "ctx_subqueries": len(decomposition.subqueries),
            },
        )

        return RetrievalEngineResult(
            query=query,
            resolved_query=retrieval_query,
            rewrite_result=rewrite_result,
            original_analysis=original_analysis,
            resolved_analysis=resolved_analysis,
            decomposition_result=decomposition,
            subquery_results=subquery_results,
            collection_id=collection_id,
            strategy=decision.strategy,
            results=merged_results,
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
            fused_hits=fused_hits,
            retrieval_overlap=retrieval_overlap,
            decision=decision,
            confidence=confidence,
            generated_answer=generated_answer,
            trace=trace,
        )

    def _retrieve_subquery(
        self,
        *,
        subquery: SubQuery,
        collection_id: str,
        metadata_filters: dict | None,
        explicit_strategy: RetrievalStrategy | None,
        top_k: int,
        corpus_size: int,
    ) -> SubQueryResult:
        sub_analysis = self._query_analyzer.analyze(subquery.query)
        if explicit_strategy is not None:
            decision = RetrievalDecision(
                strategy=explicit_strategy,
                reason="Explicit strategy requested by caller",
            )
        else:
            decision = self._adaptive_router.decide(sub_analysis)

        scope = self._scope_builder.build(
            analysis=sub_analysis,
            collection_id=collection_id,
            estimated_corpus_size=corpus_size,
            extra_filters=metadata_filters,
        )
        plan = SubQueryRetrievalPlan(
            subquery=subquery,
            strategy=decision.strategy,
            reason=decision.reason,
            top_k=top_k,
        )
        retrieval_result = self._hybrid_retriever.retrieve(
            query=subquery.query,
            collection_id=collection_id,
            strategy=plan.strategy,
            metadata_filters=dict(scope.filters),
            top_k=plan.top_k,
        )
        confidence = self._confidence_scorer.score(
            results=retrieval_result.results,
            retrieval_overlap=retrieval_result.retrieval_overlap,
            analysis=sub_analysis,
        )
        return SubQueryResult(
            subquery=subquery,
            plan=plan,
            results=retrieval_result.results,
            dense_hits=retrieval_result.dense_hits,
            sparse_hits=retrieval_result.sparse_hits,
            fused_hits=retrieval_result.fused_hits,
            retrieval_overlap=retrieval_result.retrieval_overlap,
            confidence=confidence,
            trace=retrieval_result.trace,
        )

    def _maybe_rewrite(
        self,
        *,
        query: str,
        analysis: OriginalQueryAnalysis,
        context_messages: list[Message],
    ) -> RewriteResult:
        if not analysis.rewrite_decision.should_rewrite:
            return RewriteResult(
                original_query=query,
                rewritten_query=query,
                was_rewritten=False,
                reason=analysis.rewrite_decision.reason,
            )
        return self._query_rewriter.rewrite(query, analysis, context_messages=context_messages)

    @staticmethod
    def _build_aggregate_decision(
        subquery_results: list[SubQueryResult],
        *,
        explicit_strategy: RetrievalStrategy | None,
    ) -> RetrievalDecision:
        if explicit_strategy is not None:
            return RetrievalDecision(
                strategy=explicit_strategy,
                reason="Explicit strategy requested by caller",
            )
        if not subquery_results:
            return RetrievalDecision(
                strategy=RetrievalStrategy.HYBRID,
                reason="No subquery results; default hybrid strategy",
            )

        strategies = {result.plan.strategy for result in subquery_results}
        if len(strategies) == 1:
            first = subquery_results[0].plan
            return RetrievalDecision(strategy=first.strategy, reason=first.reason)

        summary = ", ".join(
            f"{result.subquery.id}={result.plan.strategy.value}" for result in subquery_results
        )
        return RetrievalDecision(
            strategy=RetrievalStrategy.HYBRID,
            reason=f"Per-subquery routing across {len(subquery_results)} subqueries: {summary}",
        )

    def _aggregate_confidence(
        self,
        subquery_results: list[SubQueryResult],
        resolved_analysis: ResolvedQueryAnalysis,
        merged_results: list[ScoredChunk],
    ) -> ConfidenceScore:
        subquery_confidence = self._confidence_scorer.aggregate(
            [result.confidence for result in subquery_results]
        )
        if subquery_confidence is None:
            return self._confidence_scorer.score(
                results=merged_results,
                retrieval_overlap=0.0,
                analysis=resolved_analysis,
            )

        merged_score = self._confidence_scorer.score(
            results=merged_results,
            retrieval_overlap=subquery_confidence.breakdown.retrieval_overlap,
            analysis=resolved_analysis,
        )
        value = round((subquery_confidence.value + merged_score.value) / 2.0, 4)
        return ConfidenceScore(
            value=value,
            is_acceptable=value >= subquery_confidence.threshold,
            threshold=subquery_confidence.threshold,
            breakdown=merged_score.breakdown,
            weights=subquery_confidence.weights,
        )

    @staticmethod
    def _resolve_fused_hits(
        subquery_results: list[SubQueryResult],
        merged_results: list[ScoredChunk],
    ) -> list[ScoredChunk]:
        if len(subquery_results) > 1:
            return merged_results

        if len(subquery_results) == 1:
            if subquery_results[0].plan.strategy == RetrievalStrategy.HYBRID:
                return subquery_results[0].fused_hits
            return []

        return []

    @staticmethod
    def _dedupe_hits(hits: list[ScoredChunk]) -> list[ScoredChunk]:
        seen: set[str] = set()
        deduped: list[ScoredChunk] = []
        for hit in hits:
            chunk_id = hit.chunk.id
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            deduped.append(hit)
        return deduped

    @staticmethod
    def _append_trace_steps(
        *,
        trace: RetrievalTrace,
        original_analysis: OriginalQueryAnalysis,
        resolved_analysis: ResolvedQueryAnalysis,
        rewrite_result: RewriteResult,
        retrieval_query: str,
        original_analyze_ms: float,
        resolved_analyze_ms: float,
        decomposition: DecompositionResult,
        subquery_results: list[SubQueryResult],
        decision: RetrievalDecision,
        confidence: ConfidenceScore,
        merged_count: int,
        rerank_ms: float = 0.0,
        rerank_metadata: dict[str, object] | None = None,
    ) -> None:
        trace.steps.append(
            StepTrace(
                step="original_query_analysis",
                duration_ms=original_analyze_ms,
                metadata={
                    "query_type": original_analysis.query_type.value,
                    "intent": original_analysis.intent.value,
                    "should_rewrite": original_analysis.rewrite_decision.should_rewrite,
                },
            )
        )
        trace.steps.append(
            StepTrace(
                step="query_rewrite",
                duration_ms=0.0,
                metadata={
                    "was_rewritten": rewrite_result.was_rewritten,
                    "reason": rewrite_result.reason,
                    "rewritten_query": retrieval_query,
                },
            )
        )
        if rewrite_result.was_rewritten:
            trace.steps.append(
                StepTrace(
                    step="resolved_query_analysis",
                    duration_ms=resolved_analyze_ms,
                    metadata={
                        "query_type": resolved_analysis.query_type.value,
                        "intent": resolved_analysis.intent.value,
                        "should_decompose": resolved_analysis.decomposition_decision.should_decompose,
                    },
                )
            )

        trace.steps.append(
            StepTrace(
                step="query_decomposition",
                duration_ms=0.0,
                metadata={
                    "was_decomposed": decomposition.was_decomposed,
                    "reason": decomposition.reason,
                    "subquery_count": len(decomposition.subqueries),
                    "subqueries": [
                        {
                            "id": subquery.id,
                            "query": subquery.query,
                            "query_type": subquery.query_type.value,
                            "source": subquery.source.value,
                        }
                        for subquery in decomposition.subqueries
                    ],
                },
            )
        )

        for result in subquery_results:
            trace.steps.append(
                StepTrace(
                    step=f"subquery_retrieval_{result.subquery.id}",
                    duration_ms=sum(result.trace.latency_ms.values()),
                    metadata={
                        "query": result.subquery.query,
                        "strategy": result.plan.strategy.value,
                        "reason": result.plan.reason,
                        "top_k": result.plan.top_k,
                        "hits": len(result.results),
                        "confidence": result.confidence.value,
                    },
                )
            )

        trace.steps.append(
            StepTrace(
                step="subquery_merge",
                duration_ms=0.0,
                metadata={"merged_hits": merged_count, "aggregate_strategy": decision.strategy.value},
            )
        )
        if rerank_metadata is not None:
            trace.steps.append(
                StepTrace(
                    step="rerank",
                    duration_ms=rerank_ms,
                    metadata=rerank_metadata,
                )
            )
        trace.steps.append(
            StepTrace(
                step="confidence_scoring",
                duration_ms=0.0,
                metadata={"confidence": confidence.value, "acceptable": confidence.is_acceptable},
            )
        )
