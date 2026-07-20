"""Adaptive retrieval use case."""

from __future__ import annotations

from typing import Any

from adaptive_rag.application.dto.retrieval import RetrievalEngineResult, RetrievalRequest
from adaptive_rag.application.services.retrieval_engine import RetrievalEngine
from adaptive_rag.domain.models.conversation import Message
from adaptive_rag.domain.models.retrieval import RetrievalStrategy


class HybridRetrievalUseCase:
    """Execute adaptive retrieval via the RetrievalEngine."""

    def __init__(self, retrieval_engine: RetrievalEngine) -> None:
        self._engine = retrieval_engine

    def execute(
        self,
        *,
        query: str,
        collection_id: str = "default",
        strategy: RetrievalStrategy | None = None,
        metadata_filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        context_messages: list[Message] | None = None,
    ) -> RetrievalEngineResult:
        """Run adaptive retrieval and return ranked chunks with confidence."""
        return self._engine.execute(
            query=query,
            collection_id=collection_id,
            strategy=strategy,
            metadata_filters=metadata_filters,
            top_k=top_k,
            context_messages=context_messages,
        )

    def execute_request(self, request: RetrievalRequest) -> RetrievalEngineResult:
        """Run retrieval from a structured request DTO."""
        return self.execute(
            query=request.query,
            collection_id=request.collection_id,
            strategy=request.strategy,
            metadata_filters=request.metadata_filters or None,
            top_k=request.top_k,
            context_messages=request.context_messages or None,
        )
