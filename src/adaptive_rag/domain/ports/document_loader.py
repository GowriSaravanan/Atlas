"""Document loader port."""

from __future__ import annotations

from typing import Protocol

from adaptive_rag.domain.models.document import Document


class DocumentLoaderPort(Protocol):
    """Load raw documents from a source path."""

    def load(self, source_path: str) -> Document:
        """Load a document from the given source path."""
        ...
