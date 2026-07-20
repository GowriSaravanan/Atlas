"""Citation formatter port."""

from __future__ import annotations

from typing import Protocol

from adaptive_rag.domain.models.answer import GeneratedAnswer
from adaptive_rag.domain.models.retrieval import ScoredChunk


class CitationFormatterPort(Protocol):
    """Convert retrieved evidence into structured citations."""

    def format(
        self,
        answer: GeneratedAnswer,
        evidence: list[ScoredChunk],
    ) -> GeneratedAnswer:
        """Attach structured citations and formatted views to an answer."""
        ...
