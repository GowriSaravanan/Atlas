"""Query analysis domain models."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class QueryIntent(str, Enum):
    """High-level intent classification for a user query."""

    FACTUAL = "factual"
    COMPARATIVE = "comparative"
    PROCEDURAL = "procedural"
    SUMMARIZATION = "summarization"
    CHITCHAT = "chitchat"
    UNKNOWN = "unknown"


class ComplexityLevel(str, Enum):
    """Estimated complexity of a user query."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class QueryAnalysis(BaseModel):
    """Structured output from query analysis (rules + LLM)."""

    intent: QueryIntent = QueryIntent.UNKNOWN
    complexity: ComplexityLevel = ComplexityLevel.MEDIUM
    entities: list[str] = Field(default_factory=list)
    metadata_hints: dict[str, Any] = Field(default_factory=dict)
    is_multi_question: bool = False
    needs_decomposition: bool = False
    needs_pre_rewrite: bool = False
    rule_matches: list[str] = Field(default_factory=list)
