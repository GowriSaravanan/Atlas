"""Phase 4B query decomposition tests."""

from __future__ import annotations

from pathlib import Path

from adaptive_rag.api.dependencies.container import get_container
from adaptive_rag.domain.models.decomposition import SubQuerySource
from adaptive_rag.domain.models.query import QueryType
from adaptive_rag.domain.models.retrieval import RetrievalStrategy
from adaptive_rag.domain.policies.query_analyzer import QueryAnalyzer
from adaptive_rag.domain.policies.query_decomposer import RuleBasedQueryDecomposer
from adaptive_rag.domain.policies.top_k_allocator import TopKAllocator


def test_decomposer_pass_through_single_subquery() -> None:
    analyzer = QueryAnalyzer()
    decomposer = RuleBasedQueryDecomposer(analyzer)
    analysis = analyzer.analyze("What is HR-203?")

    result = decomposer.decompose("What is HR-203?", analysis)

    assert result.was_decomposed is False
    assert len(result.subqueries) == 1
    assert result.subqueries[0].id == "0"
    assert result.subqueries[0].source == SubQuerySource.PASS_THROUGH
    assert result.subqueries[0].query == "What is HR-203?"
    assert result.subqueries[0].query_type == QueryType.LOOKUP


def test_decomposer_splits_comparison_query() -> None:
    analyzer = QueryAnalyzer()
    decomposer = RuleBasedQueryDecomposer(analyzer)
    query = "Compare maternity leave and paternity leave benefits."
    analysis = analyzer.analyze(query)

    result = decomposer.decompose(query, analysis)

    assert result.was_decomposed is True
    assert len(result.subqueries) == 2
    assert {subquery.id for subquery in result.subqueries} == {"A", "B"}
    assert all(subquery.source == SubQuerySource.COMPARISON for subquery in result.subqueries)
    assert "maternity leave" in result.subqueries[0].query.lower()
    assert "paternity leave" in result.subqueries[1].query.lower()


def test_top_k_allocator_uses_weighted_budget() -> None:
    allocator = TopKAllocator()
    analyzer = QueryAnalyzer()
    decomposer = RuleBasedQueryDecomposer(analyzer)
    query = "Compare maternity leave and paternity leave benefits."
    analysis = analyzer.analyze(query)
    decomposition = decomposer.decompose(query, analysis)

    allocations = allocator.allocate(decomposition.subqueries, total_budget=10)

    assert sum(allocations.values()) == 10
    assert allocations["A"] == 5
    assert allocations["B"] == 5


def test_top_k_allocator_lookup_vs_semantic_weighting() -> None:
    allocator = TopKAllocator()
    lookup = analyzer_subquery("What is HR-203?", QueryType.LOOKUP)
    semantic = analyzer_subquery("Explain employee leave benefits in detail", QueryType.SEMANTIC)

    allocations = allocator.allocate([lookup, semantic], total_budget=10)

    assert allocations[lookup.id] == 3
    assert allocations[semantic.id] == 7


def analyzer_subquery(query: str, expected_type: QueryType):
    from adaptive_rag.domain.models.decomposition import SubQuery

    return SubQuery(
        id=query[:1],
        query=query,
        entity=None,
        source=SubQuerySource.PASS_THROUGH,
        query_type=expected_type,
        parent_query=query,
    )


def test_retrieval_engine_decomposes_comparison_with_per_subquery_routing(sample_pdf: Path) -> None:
    container = get_container()
    collection_id = "adaptive-phase4b"
    container.ingest_document_use_case.execute(source_path=str(sample_pdf), collection_id=collection_id)

    result = container.hybrid_retrieval_use_case.execute(
        query="Compare maternity leave and paternity leave benefits.",
        collection_id=collection_id,
        top_k=10,
    )

    assert result.decomposition_result is not None
    assert result.decomposition_result.was_decomposed is True
    assert len(result.subquery_results) == 2
    assert len(result.results) >= 1

    strategies = {subquery_result.plan.strategy for subquery_result in result.subquery_results}
    assert len(strategies) >= 1
    assert result.decision is not None
    assert "Per-subquery routing" in result.decision.reason or len(strategies) == 1

    step_names = [step.step for step in result.trace.steps]
    assert "query_decomposition" in step_names
    assert "subquery_retrieval_A" in step_names
    assert "subquery_retrieval_B" in step_names
    assert "subquery_merge" in step_names


def test_retrieval_engine_pass_through_still_returns_subquery_results(sample_pdf: Path) -> None:
    container = get_container()
    collection_id = "adaptive-phase4b-pass"
    container.ingest_document_use_case.execute(source_path=str(sample_pdf), collection_id=collection_id)

    result = container.hybrid_retrieval_use_case.execute(
        query="What is HR-203?",
        collection_id=collection_id,
    )

    assert result.decomposition_result is not None
    assert result.decomposition_result.was_decomposed is False
    assert len(result.decomposition_result.subqueries) == 1
    assert len(result.subquery_results) == 1
    assert result.subquery_results[0].subquery.id == "0"
    assert result.subquery_results[0].plan.strategy == RetrievalStrategy.BM25
