"""Evaluator port for RAGAS and LLM-as-Judge."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from adaptive_rag.domain.models.grounding import Citation


class EvaluationInput(BaseModel):
    """Input payload for offline RAG evaluation."""

    query: str
    answer: str
    contexts: list[str]
    ground_truth: str | None = None
    citations: list[Citation] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    """Aggregated evaluation metrics."""

    metrics: dict[str, float] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)


class EvaluatorPort(Protocol):
    """Evaluate RAG quality using RAGAS, LLM-as-Judge, or custom metrics."""

    def evaluate(self, inputs: list[EvaluationInput]) -> EvaluationResult:
        """Run batch evaluation and return aggregated metrics."""
        ...

    def judge_answer(self, query: str, answer: str, contexts: list[str]) -> EvaluationResult:
        """Score a single answer with LLM-as-Judge."""
        ...
