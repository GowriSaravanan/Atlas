"""Answer generation domain models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from adaptive_rag.domain.models.retrieval import ScoredChunk


class BuiltContext(BaseModel):
    """Evidence context prepared for answer generation."""

    context: str
    used_chunks: list[ScoredChunk] = Field(default_factory=list)
    used_chunk_ids: list[str] = Field(default_factory=list)
    estimated_tokens: int = 0
    truncated: bool = False


class GeneratedAnswer(BaseModel):
    """Structured answer output independent of LLM provider APIs."""

    answer: str
    used_chunk_ids: list[str] = Field(default_factory=list)
    model_name: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
