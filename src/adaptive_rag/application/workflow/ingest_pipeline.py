"""Document ingestion pipeline skeleton."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from adaptive_rag.application.workflow.state import IngestGraphState
from adaptive_rag.observability.logging import get_logger

logger = get_logger(__name__)


def _load_document(state: IngestGraphState) -> dict:
    """Placeholder: load document (Phase 1)."""
    return {"status": "pending"}


def _chunk_document(state: IngestGraphState) -> dict:
    """Placeholder: chunk document (Phase 1)."""
    return {"status": "pending"}


def _index_document(state: IngestGraphState) -> dict:
    """Placeholder: index document (Phase 1)."""
    return {"status": "completed", "chunk_count": 0}


def build_ingest_graph() -> StateGraph:
    """Build the ingestion pipeline graph (skeleton)."""
    graph: StateGraph = StateGraph(IngestGraphState)

    graph.add_node("load_document", _load_document)
    graph.add_node("chunk_document", _chunk_document)
    graph.add_node("index_document", _index_document)

    graph.add_edge(START, "load_document")
    graph.add_edge("load_document", "chunk_document")
    graph.add_edge("chunk_document", "index_document")
    graph.add_edge("index_document", END)

    logger.debug("Ingest graph skeleton built")
    return graph


def compile_ingest_graph():
    """Compile the ingestion pipeline graph."""
    return build_ingest_graph().compile()
