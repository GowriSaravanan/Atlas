"""LangGraph node registry — maps node IDs to handler callables (Phase 1+)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from adaptive_rag.application.workflow.state import RAGGraphState


NodeHandler = Callable[[RAGGraphState], dict[str, Any]]


def passthrough_node(state: RAGGraphState) -> dict[str, Any]:
    """Placeholder node that returns state unchanged."""
    return {}


NODE_REGISTRY: dict[str, NodeHandler] = {
    "resolve_context": passthrough_node,
    "analyze_query": passthrough_node,
    "pre_rewrite": passthrough_node,
    "extract_metadata": passthrough_node,
    "metadata_filter": passthrough_node,
    "decompose": passthrough_node,
    "adaptive_route": passthrough_node,
    "retrieve_bm25": passthrough_node,
    "retrieve_dense": passthrough_node,
    "fuse": passthrough_node,
    "rerank": passthrough_node,
    "score_retrieval_confidence": passthrough_node,
    "escalate": passthrough_node,
    "merge_chunks": passthrough_node,
    "generate_answer": passthrough_node,
    "grounding_validator": passthrough_node,
    "citation_verifier": passthrough_node,
    "hallucination_guard": passthrough_node,
    "enrich_confidence": passthrough_node,
    "format_response": passthrough_node,
}
