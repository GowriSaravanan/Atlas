"""Weighted adaptive router tests."""

from __future__ import annotations

import pytest

from adaptive_rag.domain.models.query import QueryType
from adaptive_rag.domain.models.retrieval import RetrievalStrategy
from adaptive_rag.domain.policies.adaptive_router import AdaptiveRouter
from adaptive_rag.domain.policies.query_analyzer import QueryAnalyzer
from adaptive_rag.domain.policies.query_decomposer import RuleBasedQueryDecomposer
from adaptive_rag.domain.policies.query_rewriter import RuleBasedQueryRewriter


@pytest.fixture
def router() -> AdaptiveRouter:
    return AdaptiveRouter()


@pytest.fixture
def analyzer() -> QueryAnalyzer:
    return QueryAnalyzer()


def test_router_selects_bm25_for_single_policy_lookup(router: AdaptiveRouter, analyzer: QueryAnalyzer) -> None:
    decision = router.decide(analyzer.analyze("What is HR-203?"))
    assert decision.strategy == RetrievalStrategy.BM25
    assert "policy_identifier_lookup" in decision.reason


def test_router_selects_hybrid_for_comparison_with_policy_ids(
    router: AdaptiveRouter, analyzer: QueryAnalyzer
) -> None:
    analysis = analyzer.analyze("Compare HR-203 with HR-105")
    decision = router.decide(analysis)

    assert analysis.query_type == QueryType.COMPARISON
    assert decision.strategy == RetrievalStrategy.HYBRID
    assert "comparison_intent" in decision.reason


def test_router_selects_hybrid_for_maternity_benefits_query(
    router: AdaptiveRouter, analyzer: QueryAnalyzer
) -> None:
    decision = router.decide(analyzer.analyze("Explain employee maternity leave benefits in detail"))
    assert decision.strategy == RetrievalStrategy.HYBRID
    assert "hr_policy_domain" in decision.reason or "semantic_query_type" in decision.reason


def test_router_selects_hybrid_for_sick_leave_count_query(
    router: AdaptiveRouter, analyzer: QueryAnalyzer
) -> None:
    decision = router.decide(analyzer.analyze("How many sick leave days are allowed?"))
    assert decision.strategy == RetrievalStrategy.HYBRID


def test_router_selects_hybrid_for_annual_leave_policy_query(
    router: AdaptiveRouter, analyzer: QueryAnalyzer
) -> None:
    decision = router.decide(analyzer.analyze("What is the annual leave policy for new hires?"))
    assert decision.strategy == RetrievalStrategy.HYBRID


def test_router_reason_includes_score_breakdown(router: AdaptiveRouter, analyzer: QueryAnalyzer) -> None:
    decision = router.decide(analyzer.analyze("Compare maternity leave and paternity leave benefits."))
    assert "score=" in decision.reason
    assert decision.strategy == RetrievalStrategy.HYBRID


def test_query_rewriter_avoids_duplicate_policy_suffix() -> None:
    rewriter = RuleBasedQueryRewriter()
    analysis = QueryAnalyzer().analyze("Tell me about sick leave policy")

    result = rewriter.rewrite("Tell me about sick leave policy", analysis)

    assert result.rewritten_query == "What is the sick leave policy?"
    assert "policy policy" not in result.rewritten_query.lower()


def test_decomposer_splits_compare_versus_pattern() -> None:
    analyzer = QueryAnalyzer()
    decomposer = RuleBasedQueryDecomposer(analyzer)
    query = "Compare HR-203 versus HR-105 leave allowances"
    result = decomposer.decompose(query, analyzer.analyze(query))

    assert result.was_decomposed is True
    assert [subquery.query for subquery in result.subqueries] == [
        "What is HR-203?",
        "What is HR-105?",
    ]
