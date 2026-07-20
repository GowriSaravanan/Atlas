"""Answer generator port."""

from __future__ import annotations

from typing import Protocol

from adaptive_rag.domain.models.answer import GeneratedAnswer
from adaptive_rag.domain.models.retrieval import ScoredChunk


class AnswerGeneratorPort(Protocol):
    """Convert retrieved evidence into a grounded answer."""

    def generate(self, query: str, evidence: list[ScoredChunk]) -> GeneratedAnswer:
        """Generate an answer from reranked retrieval evidence."""
        ...
