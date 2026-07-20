"""Document ingestion pipeline."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from adaptive_rag.application.workflow.nodes.ingest_nodes import IngestNodeContext, build_node_handlers
from adaptive_rag.application.workflow.state import IngestGraphState
from adaptive_rag.observability.logging import get_logger

logger = get_logger(__name__)


def _route_after_load(state: IngestGraphState) -> str:
    if state.get("status") == "failed":
        return "failed"
    return "extract_metadata"


def _route_after_metadata(state: IngestGraphState) -> str:
    if state.get("status") == "failed":
        return "failed"
    return "chunk_document"


def _route_after_chunk(state: IngestGraphState) -> str:
    if state.get("status") == "failed":
        return "failed"
    return "index_document"


def _mark_failed(_state: IngestGraphState) -> dict:
    return {"status": "failed"}


def build_ingest_graph(context: IngestNodeContext) -> StateGraph:
    """Build the ingestion pipeline graph."""
    graph: StateGraph = StateGraph(IngestGraphState)
    handlers = build_node_handlers(context)

    for node_id, handler in handlers.items():
        graph.add_node(node_id, handler)

    graph.add_node("failed", _mark_failed)

    graph.add_edge(START, "load_document")
    graph.add_conditional_edges(
        "load_document",
        _route_after_load,
        {"extract_metadata": "extract_metadata", "failed": "failed"},
    )
    graph.add_conditional_edges(
        "extract_metadata",
        _route_after_metadata,
        {"chunk_document": "chunk_document", "failed": "failed"},
    )
    graph.add_conditional_edges(
        "chunk_document",
        _route_after_chunk,
        {"index_document": "index_document", "failed": "failed"},
    )
    graph.add_edge("index_document", END)
    graph.add_edge("failed", END)

    logger.debug("Ingest graph built with metadata extraction and indexing")
    return graph


def compile_ingest_graph(context: IngestNodeContext):
    """Compile the ingestion pipeline graph."""
    return build_ingest_graph(context).compile()
