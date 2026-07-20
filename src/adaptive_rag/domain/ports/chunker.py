"""Chunker port."""

from __future__ import annotations

from typing import Protocol

from adaptive_rag.domain.models.document import Chunk, Document


class ChunkerPort(Protocol):
    """Split documents into chunks."""

    def chunk(self, document: Document) -> list[Chunk]:
        """Chunk a document into retrievable units."""
        ...
