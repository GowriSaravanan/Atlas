"""Ingestion workflow node handlers."""

from __future__ import annotations

from dataclasses import dataclass

from adaptive_rag.application.workflow.state import IngestGraphState
from adaptive_rag.domain.policies.adaptive_chunker import AdaptiveChunker
from adaptive_rag.domain.policies.document_metadata_extractor import DocumentMetadataExtractor
from adaptive_rag.domain.ports.document_loader import DocumentLoaderPort
from adaptive_rag.domain.ports.index_registry import IndexRegistryPort
from adaptive_rag.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class IngestNodeContext:
    """Dependencies injected into ingestion workflow nodes."""

    document_loader: DocumentLoaderPort
    metadata_extractor: DocumentMetadataExtractor
    chunker: AdaptiveChunker
    index_registry: IndexRegistryPort


def load_document_node(context: IngestNodeContext, state: IngestGraphState) -> dict:
    """Load source document from disk."""
    try:
        document = context.document_loader.load(state["source_path"])
        return {
            "document": document,
            "document_id": document.id,
            "status": "pending",
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 — surface pipeline failures in state
        logger.exception("Failed to load document", extra={"ctx_source_path": state["source_path"]})
        return {"status": "failed", "error": str(exc)}


def extract_metadata_node(context: IngestNodeContext, state: IngestGraphState) -> dict:
    """Enrich document metadata."""
    document = state.get("document")
    if document is None:
        return {"status": "failed", "error": "Document missing before metadata extraction"}

    enriched = context.metadata_extractor.apply_to_document(document)
    return {"document": enriched}


def chunk_document_node(context: IngestNodeContext, state: IngestGraphState) -> dict:
    """Chunk document using adaptive chunking policy."""
    document = state.get("document")
    if document is None:
        return {"status": "failed", "error": "Document missing before chunking"}

    chunks = context.chunker.chunk(document)
    chunks = context.metadata_extractor.apply_to_chunks(chunks)
    return {"chunks": chunks, "chunk_count": len(chunks)}


def index_document_node(context: IngestNodeContext, state: IngestGraphState) -> dict:
    """Embed and index chunks via the index registry port."""
    chunks = state.get("chunks") or []
    if not chunks:
        return {"status": "failed", "error": "No chunks produced during ingestion"}

    collection_id = state.get("collection_id", "default")
    context.index_registry.index_chunks(collection_id, chunks)

    logger.info(
        "Indexed document",
        extra={
            "ctx_collection_id": collection_id,
            "ctx_document_id": state.get("document_id"),
            "ctx_chunk_count": len(chunks),
        },
    )
    return {"status": "completed", "chunk_count": len(chunks), "error": None}


def build_node_handlers(context: IngestNodeContext) -> dict[str, object]:
    """Create bound node handlers for LangGraph registration."""
    return {
        "load_document": lambda state: load_document_node(context, state),
        "extract_metadata": lambda state: extract_metadata_node(context, state),
        "chunk_document": lambda state: chunk_document_node(context, state),
        "index_document": lambda state: index_document_node(context, state),
    }
