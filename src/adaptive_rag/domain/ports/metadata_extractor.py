"""Metadata extractor port."""

from __future__ import annotations

from typing import Any, Protocol

from adaptive_rag.domain.models.document import Chunk, Document
from adaptive_rag.domain.models.query import QueryAnalysis


class MetadataExtractorPort(Protocol):
    """Extract metadata from documents and queries."""

    def extract_document_metadata(self, document: Document) -> dict[str, Any]:
        """Extract structured metadata from a document."""
        ...

    def extract_chunk_metadata(self, chunk: Chunk) -> dict[str, Any]:
        """Extract or enrich metadata for a chunk."""
        ...

    def extract_query_metadata(self, analysis: QueryAnalysis) -> dict[str, Any]:
        """Derive search metadata filters from query analysis."""
        ...
