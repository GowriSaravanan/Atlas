"""LangGraph workflow state definitions."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from adaptive_rag.domain.models.confidence import ConfidenceScore
from adaptive_rag.domain.models.conversation import Message
from adaptive_rag.domain.models.grounding import Citation
from adaptive_rag.domain.models.query import QueryAnalysis
from adaptive_rag.domain.models.retrieval import (
    RetrievalStrategy,
    ScoredChunk,
    SearchScope,
    SubqueryState,
)
from adaptive_rag.domain.models.trace import RetrievalTrace


class RAGGraphState(TypedDict, total=False):
    """Typed state for the query RAG LangGraph workflow."""

    # Input
    raw_query: str
    conversation_id: str

    # Conversation
    context_window: list[Message]
    context_summary: str | None

    # Analysis
    analysis: QueryAnalysis | None
    resolved_query: str | None

    # Metadata + scope
    extracted_metadata: dict[str, Any]
    search_scope: SearchScope | None

    # Decomposition
    is_decomposed: bool
    subqueries: list[SubqueryState]

    # Single-query retrieval
    strategy: RetrievalStrategy | None
    sparse_hits: list[ScoredChunk]
    dense_hits: list[ScoredChunk]

    # Unified retrieval output
    candidate_chunks: list[ScoredChunk]
    reranked_chunks: list[ScoredChunk]

    # Confidence
    global_confidence: ConfidenceScore | None

    # Escalation
    escalation_level: int
    max_escalations: int

    # Generation
    answer: str | None
    citations: list[Citation]
    answer_mode: Literal["full", "partial", "insufficient"]
    regen_attempted: bool

    # Observability
    trace: RetrievalTrace


class IngestGraphState(TypedDict, total=False):
    """Typed state for the ingestion pipeline."""

    source_path: str
    collection_id: str
    document_id: str | None
    chunk_count: int
    status: Literal["pending", "completed", "failed"]
    error: str | None


def initial_rag_state(*, raw_query: str, conversation_id: str) -> RAGGraphState:
    """Build initial query workflow state."""
    return RAGGraphState(
        raw_query=raw_query,
        conversation_id=conversation_id,
        context_window=[],
        context_summary=None,
        analysis=None,
        resolved_query=None,
        extracted_metadata={},
        search_scope=None,
        is_decomposed=False,
        subqueries=[],
        strategy=None,
        sparse_hits=[],
        dense_hits=[],
        candidate_chunks=[],
        reranked_chunks=[],
        global_confidence=None,
        escalation_level=0,
        max_escalations=3,
        answer=None,
        citations=[],
        answer_mode="full",
        regen_attempted=False,
        trace=RetrievalTrace(),
    )


def initial_ingest_state(*, source_path: str, collection_id: str = "default") -> IngestGraphState:
    """Build initial ingestion workflow state."""
    return IngestGraphState(
        source_path=source_path,
        collection_id=collection_id,
        document_id=None,
        chunk_count=0,
        status="pending",
        error=None,
    )
