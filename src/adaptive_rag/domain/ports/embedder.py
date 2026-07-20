"""Embedder port."""

from __future__ import annotations

from typing import Protocol

from adaptive_rag.domain.models.document import Chunk


class EmbedderPort(Protocol):
    """Generate vector embeddings for text chunks."""

    @property
    def model_name(self) -> str:
        """Return the embedding model identifier."""
        ...

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        ...

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        ...

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        """Embed a batch of chunks."""
        ...
