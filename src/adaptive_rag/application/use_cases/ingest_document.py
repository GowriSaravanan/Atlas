"""Ingest document use case."""

from __future__ import annotations

from adaptive_rag.application.dto.responses import IngestDocumentResponse
from adaptive_rag.application.workflow.ingest_pipeline import compile_ingest_graph
from adaptive_rag.application.workflow.state import initial_ingest_state
from adaptive_rag.observability.logging import get_logger

logger = get_logger(__name__)


class IngestDocumentUseCase:
    """Orchestrate document ingestion via the ingest graph."""

    def __init__(self) -> None:
        self._graph = compile_ingest_graph()

    def execute(self, *, source_path: str, collection_id: str = "default") -> IngestDocumentResponse:
        """Run the ingestion pipeline for a source document."""
        state = initial_ingest_state(source_path=source_path, collection_id=collection_id)
        result = self._graph.invoke(state)
        logger.info("Ingestion completed", extra={"ctx_status": result.get("status")})
        return IngestDocumentResponse(
            document_id=result.get("document_id") or "",
            chunk_count=result.get("chunk_count", 0),
            status=result.get("status", "failed"),
            message=result.get("error") or "Ingestion skeleton executed",
        )
