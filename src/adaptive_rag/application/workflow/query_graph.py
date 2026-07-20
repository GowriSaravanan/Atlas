"""Query RAG LangGraph workflow skeleton."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from adaptive_rag.application.workflow.nodes.registry import NODE_REGISTRY
from adaptive_rag.application.workflow.state import RAGGraphState
from adaptive_rag.observability.logging import get_logger

logger = get_logger(__name__)


def build_query_graph() -> StateGraph:
    """Build the query RAG workflow graph (skeleton — nodes are passthrough)."""
    graph: StateGraph = StateGraph(RAGGraphState)

    for node_id, handler in NODE_REGISTRY.items():
        graph.add_node(node_id, handler)

    # Phase 0: linear skeleton through core stages
    graph.add_edge(START, "resolve_context")
    graph.add_edge("resolve_context", "analyze_query")
    graph.add_edge("analyze_query", "extract_metadata")
    graph.add_edge("extract_metadata", "metadata_filter")
    graph.add_edge("metadata_filter", "adaptive_route")
    graph.add_edge("adaptive_route", "retrieve_bm25")
    graph.add_edge("retrieve_bm25", "retrieve_dense")
    graph.add_edge("retrieve_dense", "fuse")
    graph.add_edge("fuse", "rerank")
    graph.add_edge("rerank", "score_retrieval_confidence")
    graph.add_edge("score_retrieval_confidence", "generate_answer")
    graph.add_edge("generate_answer", "grounding_validator")
    graph.add_edge("grounding_validator", "citation_verifier")
    graph.add_edge("citation_verifier", "hallucination_guard")
    graph.add_edge("hallucination_guard", "enrich_confidence")
    graph.add_edge("enrich_confidence", "format_response")
    graph.add_edge("format_response", END)

    logger.debug("Query graph skeleton built with %d nodes", len(NODE_REGISTRY))
    return graph


def compile_query_graph():
    """Compile the query workflow graph."""
    return build_query_graph().compile()
