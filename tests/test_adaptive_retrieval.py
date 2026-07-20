"""Phase 3–4 adaptive retrieval tests."""

from __future__ import annotations

from pathlib import Path

from adaptive_rag.api.dependencies.container import get_container
from adaptive_rag.config.settings import get_settings
from adaptive_rag.domain.models.conversation import Message, MessageRole
from adaptive_rag.domain.models.query import QueryIntent, QueryType
from adaptive_rag.domain.models.retrieval import RetrievalStrategy
from adaptive_rag.domain.policies.adaptive_router import AdaptiveRouter
from adaptive_rag.domain.policies.confidence import ConfidenceScorer
from adaptive_rag.domain.policies.metadata_scope_builder import MetadataScopeBuilder
from adaptive_rag.domain.policies.query_analyzer import QueryAnalyzer
from adaptive_rag.domain.policies.query_rewriter import RuleBasedQueryRewriter


def test_query_analyzer_detects_policy_id() -> None:
    analyzer = QueryAnalyzer()
    analysis = analyzer.analyze("What is HR-203?")

    assert analysis.intent == QueryIntent.LOOKUP
    assert analysis.query_type == QueryType.LOOKUP
    assert "HR-203" in analysis.entities
    assert analysis.metadata_scope.get("policy_id") == "HR-203"
    assert "policy_id" in analysis.rule_matches
    assert analysis.complexity.value == "low"


def test_query_analyzer_detects_department_and_year() -> None:
    analyzer = QueryAnalyzer()
    analysis = analyzer.analyze("HR leave policy for 2024")

    assert analysis.metadata_scope.get("department") == "HR"
    assert analysis.metadata_scope.get("year") == "2024"


def test_adaptive_router_selects_bm25_for_policy_lookup() -> None:
    analyzer = QueryAnalyzer()
    router = AdaptiveRouter()
    analysis = analyzer.analyze("What is HR-203?")
    decision = router.decide(analysis)

    assert decision.strategy == RetrievalStrategy.BM25
    assert "policy_identifier" in decision.reason


def test_metadata_scope_builder_merges_analysis() -> None:
    analyzer = QueryAnalyzer()
    builder = MetadataScopeBuilder()
    analysis = analyzer.analyze("HR leave policy 2024")
    scope = builder.build(analysis=analysis, collection_id="default", estimated_corpus_size=10)

    assert scope.filters["collection_id"] == "default"
    assert scope.filters["department"] == "HR"
    assert scope.filters["year"] == "2024"


def test_confidence_scorer_produces_breakdown() -> None:
    from adaptive_rag.domain.models.document import Chunk
    from adaptive_rag.domain.models.retrieval import ScoredChunk

    settings = get_settings()
    scorer = ConfidenceScorer(settings.retrieval)
    analysis = QueryAnalyzer().analyze("HR leave policy")
    chunk = Chunk(
        id="c1",
        document_id="d1",
        content="HR leave",
        metadata={"department": "HR", "collection_id": "default"},
    )
    results = [ScoredChunk(chunk=chunk, score=0.85, source="rrf", rank=1)]

    score = scorer.score(results=results, retrieval_overlap=0.5, analysis=analysis)

    assert 0.0 <= score.value <= 1.0
    assert score.breakdown.retrieval_overlap == 0.5
    assert score.breakdown.metadata_match > 0.0
    assert "retrieval_overlap" in score.weights


def test_retrieval_engine_adaptive_routing(sample_pdf: Path) -> None:
    container = get_container()
    collection_id = "adaptive-phase3"
    container.ingest_document_use_case.execute(source_path=str(sample_pdf), collection_id=collection_id)

    policy_result = container.hybrid_retrieval_use_case.execute(
        query="What is HR-203?",
        collection_id=collection_id,
    )
    assert policy_result.decision is not None
    assert policy_result.decision.strategy == RetrievalStrategy.BM25
    assert policy_result.original_analysis is not None
    assert policy_result.resolved_analysis is not None
    assert policy_result.analysis is policy_result.resolved_analysis
    assert policy_result.confidence is not None
    assert policy_result.trace.original_analysis is not None
    assert policy_result.trace.resolved_analysis is not None

    factual_result = container.hybrid_retrieval_use_case.execute(
        query="Explain employee leave benefits",
        collection_id=collection_id,
    )
    assert factual_result.decision.strategy in (
        RetrievalStrategy.DENSE,
        RetrievalStrategy.HYBRID,
    )


def test_rewrite_decision_for_follow_up_query() -> None:
    analysis = QueryAnalyzer().analyze("What about maternity leave?")

    assert analysis.query_type == QueryType.AMBIGUOUS
    assert analysis.rewrite_decision.should_rewrite is True
    assert analysis.rewrite_decision.reason
    assert analysis.needs_pre_rewrite is True


def test_decomposition_not_triggered_for_same_topic_compound_question() -> None:
    analysis = QueryAnalyzer().analyze(
        "What is the annual leave policy and how many days can I carry over?"
    )

    assert analysis.decomposition_decision.should_decompose is False
    assert "single policy topic" in analysis.decomposition_decision.reason
    assert analysis.needs_decomposition is False


def test_decomposition_triggered_for_comparison() -> None:
    analysis = QueryAnalyzer().analyze("Compare maternity leave and paternity leave benefits.")

    assert analysis.query_type == QueryType.COMPARISON
    assert analysis.decomposition_decision.should_decompose is True
    assert analysis.needs_decomposition is True


def test_query_rewriter_expands_follow_up() -> None:
    analyzer = QueryAnalyzer()
    rewriter = RuleBasedQueryRewriter()
    analysis = analyzer.analyze("What about maternity leave?")

    result = rewriter.rewrite("What about maternity leave?", analysis)

    assert result.was_rewritten is True
    assert result.rewritten_query == "What is the maternity leave policy?"
    assert "follow-up" in result.reason.lower() or analysis.rewrite_decision.reason


def test_query_rewriter_skips_when_not_required() -> None:
    analyzer = QueryAnalyzer()
    rewriter = RuleBasedQueryRewriter()
    analysis = analyzer.analyze("What is HR-203?")

    result = rewriter.rewrite("What is HR-203?", analysis)

    assert result.was_rewritten is False
    assert result.rewritten_query == "What is HR-203?"


def test_retrieval_engine_rewrites_follow_up_before_retrieval(sample_pdf: Path) -> None:
    container = get_container()
    collection_id = "adaptive-phase4a"
    container.ingest_document_use_case.execute(source_path=str(sample_pdf), collection_id=collection_id)

    result = container.hybrid_retrieval_use_case.execute(
        query="What about maternity leave?",
        collection_id=collection_id,
    )

    assert result.rewrite_result is not None
    assert result.rewrite_result.was_rewritten is True
    assert result.resolved_query == "What is the maternity leave policy?"
    assert result.original_analysis is not None
    assert result.resolved_analysis is not None
    assert result.original_analysis.query_type == QueryType.AMBIGUOUS
    assert result.resolved_analysis.query_type == QueryType.FACTUAL
    assert result.trace.resolved_query == result.resolved_query
    assert result.trace.steps[0].step == "original_query_analysis"
    assert result.trace.steps[1].step == "query_rewrite"
    assert result.trace.steps[1].metadata["was_rewritten"] is True
    assert result.trace.steps[2].step == "resolved_query_analysis"


def test_retrieval_engine_uses_context_for_pronoun_rewrite() -> None:
    analyzer = QueryAnalyzer()
    rewriter = RuleBasedQueryRewriter()
    analysis = analyzer.analyze("What about it?")
    context = [
        Message(role=MessageRole.USER, content="Tell me about maternity leave policy"),
    ]

    result = rewriter.rewrite("What about it?", analysis, context_messages=context)

    assert result.was_rewritten is True
    assert "maternity leave" in result.rewritten_query.lower()

