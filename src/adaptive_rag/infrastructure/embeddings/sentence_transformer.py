"""Sentence Transformers embedding adapter."""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from adaptive_rag.config.settings import EmbeddingSettings
from adaptive_rag.domain.models.document import Chunk
from adaptive_rag.domain.ports.embedder import EmbedderPort
from adaptive_rag.observability.logging import get_logger

logger = get_logger(__name__)


class SentenceTransformerEmbedder(EmbedderPort):
    """Embedder backed by Sentence Transformers."""

    def __init__(self, settings: EmbeddingSettings) -> None:
        self._settings = settings
        self._model = self._load_model(settings.model_name)

    @property
    def model_name(self) -> str:
        return self._settings.model_name

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=self._settings.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        return self.embed_texts([chunk.content for chunk in chunks])

    @staticmethod
    @lru_cache(maxsize=2)
    def _load_model(model_name: str) -> SentenceTransformer:
        logger.info("Loading embedding model", extra={"ctx_model_name": model_name})
        return SentenceTransformer(model_name)
