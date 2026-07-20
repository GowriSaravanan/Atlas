"""Deterministic fake embedder for tests."""

from __future__ import annotations

import hashlib
import math

from adaptive_rag.domain.models.document import Chunk
from adaptive_rag.domain.ports.embedder import EmbedderPort


class FakeEmbedder(EmbedderPort):
    """Lightweight deterministic embedder for unit tests."""

    def __init__(self, *, dimension: int = 384, model_name: str = "fake-embedder") -> None:
        self._dimension = dimension
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._embed_text(query)

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        return self.embed_texts([chunk.content for chunk in chunks])

    def _embed_text(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values: list[float] = []
        while len(values) < self._dimension:
            for byte in digest:
                values.append((byte / 255.0) * 2.0 - 1.0)
                if len(values) >= self._dimension:
                    break
            digest = hashlib.sha256(digest).digest()

        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]
