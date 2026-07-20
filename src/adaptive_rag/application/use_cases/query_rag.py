"""Query RAG use case."""

from __future__ import annotations

from adaptive_rag.application.dto.responses import RAGResponse
from adaptive_rag.application.workflow.query_graph import compile_query_graph
from adaptive_rag.application.workflow.state import initial_rag_state
from adaptive_rag.config.settings import get_settings
from adaptive_rag.observability.logging import get_logger

logger = get_logger(__name__)


class QueryRAGUseCase:
    """Orchestrate RAG query execution via the query graph."""

    def __init__(self) -> None:
        self._graph = compile_query_graph()

    def execute(self, *, query: str, conversation_id: str = "default") -> RAGResponse:
        """Run the query RAG workflow."""
        state = initial_rag_state(
            raw_query=query,
            conversation_id=conversation_id,
            max_escalations=get_settings().retrieval.max_escalations,
        )
        result = self._graph.invoke(state)
        logger.info("Query workflow completed", extra={"ctx_query": query})
        return RAGResponse(
            answer=result.get("answer") or "",
            answer_mode=result.get("answer_mode", "insufficient"),
            citations=result.get("citations") or [],
            global_confidence=result.get("global_confidence"),
            strategy_used=result.get("strategy"),
            subquery_summaries=[],
            escalation_count=result.get("escalation_level", 0),
            trace=result.get("trace") or state["trace"],
        )
