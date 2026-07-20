"""Adaptive retrieval routing policy."""

from __future__ import annotations

from dataclasses import dataclass

from adaptive_rag.domain.models.query import (
    ComplexityLevel,
    QueryAnalysis,
    QueryIntent,
    QueryType,
    RetrievalDecision,
)
from adaptive_rag.domain.models.retrieval import RetrievalStrategy


@dataclass(frozen=True)
class RoutingSignal:
    """Weighted contribution toward a retrieval strategy."""

    name: str
    strategy: RetrievalStrategy
    weight: float


class AdaptiveRouter:
    """Select retrieval strategy using a weighted, explainable scoring model."""

    def decide(self, analysis: QueryAnalysis) -> RetrievalDecision:
        """Return the highest-scoring strategy with an auditable signal breakdown."""
        signals = self._collect_signals(analysis)
        scores = self._score_strategies(signals)
        strategy = self._select_strategy(scores)
        reason = self._build_reason(strategy, scores, signals)
        return RetrievalDecision(strategy=strategy, reason=reason)

    def _collect_signals(self, analysis: QueryAnalysis) -> list[RoutingSignal]:
        signals: list[RoutingSignal] = []

        if analysis.intent == QueryIntent.CHITCHAT or analysis.query_type == QueryType.CONVERSATIONAL:
            signals.append(RoutingSignal("conversational", RetrievalStrategy.BM25, 5.0))

        if analysis.query_type == QueryType.COMPARISON or analysis.intent == QueryIntent.COMPARATIVE:
            signals.extend(
                [
                    RoutingSignal("comparison_intent", RetrievalStrategy.HYBRID, 6.0),
                    RoutingSignal("comparison_keyword_support", RetrievalStrategy.BM25, 1.5),
                ]
            )

        policy_ids = [entity for entity in analysis.entities if "-" in entity]
        if "policy_id" in analysis.rule_matches:
            if analysis.query_type == QueryType.COMPARISON or len(policy_ids) > 1:
                signals.extend(
                    [
                        RoutingSignal("multi_policy_comparison", RetrievalStrategy.HYBRID, 4.0),
                        RoutingSignal("policy_identifier", RetrievalStrategy.BM25, 2.0),
                    ]
                )
            else:
                signals.append(RoutingSignal("policy_identifier_lookup", RetrievalStrategy.BM25, 6.0))

        if analysis.query_type == QueryType.LOOKUP:
            signals.append(RoutingSignal("lookup_query_type", RetrievalStrategy.BM25, 4.0))

        if analysis.query_type in (QueryType.SEMANTIC, QueryType.MULTI_HOP):
            signals.extend(
                [
                    RoutingSignal("semantic_query_type", RetrievalStrategy.HYBRID, 4.0),
                    RoutingSignal("semantic_dense_support", RetrievalStrategy.DENSE, 2.0),
                ]
            )

        if analysis.intent in (QueryIntent.SUMMARIZATION, QueryIntent.PROCEDURAL):
            signals.append(RoutingSignal("summarization_or_procedural", RetrievalStrategy.HYBRID, 4.0))

        if analysis.metadata_scope:
            signals.append(RoutingSignal("metadata_scope", RetrievalStrategy.HYBRID, 3.0))

        if analysis.complexity == ComplexityLevel.HIGH:
            signals.append(RoutingSignal("high_complexity", RetrievalStrategy.HYBRID, 2.5))
        elif analysis.complexity == ComplexityLevel.MEDIUM:
            signals.append(RoutingSignal("medium_complexity", RetrievalStrategy.HYBRID, 1.5))

        if analysis.query_type == QueryType.AMBIGUOUS:
            signals.append(RoutingSignal("ambiguous_query", RetrievalStrategy.HYBRID, 2.0))

        if analysis.intent == QueryIntent.FACTUAL:
            signals.extend(
                [
                    RoutingSignal("factual_intent", RetrievalStrategy.HYBRID, 2.5),
                    RoutingSignal("factual_dense_support", RetrievalStrategy.DENSE, 1.0),
                ]
            )

        if "hr_policy_domain" in analysis.rule_matches:
            signals.append(RoutingSignal("hr_policy_domain", RetrievalStrategy.HYBRID, 2.0))

        if "short_keyword_query" in analysis.rule_matches and analysis.query_type == QueryType.FACTUAL:
            signals.append(RoutingSignal("short_keyword_query", RetrievalStrategy.DENSE, 4.0))

        if analysis.is_multi_question and analysis.query_type != QueryType.COMPARISON:
            signals.append(RoutingSignal("multi_question", RetrievalStrategy.HYBRID, 2.0))

        if not signals:
            signals.append(RoutingSignal("default_fallback", RetrievalStrategy.HYBRID, 1.0))

        return signals

    @staticmethod
    def _score_strategies(signals: list[RoutingSignal]) -> dict[RetrievalStrategy, float]:
        scores = {
            RetrievalStrategy.BM25: 0.0,
            RetrievalStrategy.DENSE: 0.0,
            RetrievalStrategy.HYBRID: 0.0,
        }
        for signal in signals:
            scores[signal.strategy] += signal.weight
        return scores

    @staticmethod
    def _select_strategy(scores: dict[RetrievalStrategy, float]) -> RetrievalStrategy:
        tie_break = {
            RetrievalStrategy.HYBRID: 3,
            RetrievalStrategy.DENSE: 2,
            RetrievalStrategy.BM25: 1,
        }
        return max(scores, key=lambda strategy: (scores[strategy], tie_break[strategy]))

    @staticmethod
    def _build_reason(
        strategy: RetrievalStrategy,
        scores: dict[RetrievalStrategy, float],
        signals: list[RoutingSignal],
    ) -> str:
        contributing = [signal for signal in signals if signal.strategy == strategy]
        contributing.sort(key=lambda signal: signal.weight, reverse=True)
        breakdown = ", ".join(f"{signal.name}+{signal.weight:g}" for signal in contributing[:4])
        score = scores[strategy]
        return f"{strategy.value} selected (score={score:g}): {breakdown}"
