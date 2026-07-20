"""Query rewriter port."""

from __future__ import annotations

from typing import Protocol

from adaptive_rag.domain.models.conversation import Message
from adaptive_rag.domain.models.query import QueryAnalysis, RewriteResult


class QueryRewriterPort(Protocol):
    """Rewrite ambiguous or context-dependent queries into standalone retrieval queries."""

    def rewrite(
        self,
        query: str,
        analysis: QueryAnalysis,
        *,
        context_messages: list[Message] | None = None,
    ) -> RewriteResult:
        """Return a rewrite result with the resolved query when applicable."""
        ...
