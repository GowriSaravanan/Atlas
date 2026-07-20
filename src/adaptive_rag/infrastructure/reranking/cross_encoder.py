"""Cross-encoder reranker backed by Sentence Transformers."""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

from adaptive_rag.config.settings import RerankerSettings
from adaptive_rag.domain.models.retrieval import ScoredChunk
from adaptive_rag.domain.ports.reranker import RerankerPort
from adaptive_rag.observability.logging import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker(RerankerPort):
    """Rerank candidate chunks with a cross-encoder relevance model."""

    def __init__(self, settings: RerankerSettings) -> None:
        self._settings = settings
        self._model = self._load_model(settings.model_name)

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int,
    ) -> list[ScoredChunk]:
        if not candidates:
            return []

        pairs = [(query, candidate.chunk.content) for candidate in candidates]
        raw_scores = self._model.predict(
            pairs,
            batch_size=self._settings.batch_size,
            show_progress_bar=False,
        )
        scored_pairs = sorted(
            zip(candidates, raw_scores, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        limit = max(1, min(top_k, len(scored_pairs)))
        reranked: list[ScoredChunk] = []
        for rank, (candidate, score) in enumerate(scored_pairs[:limit], start=1):
            reranked.append(
                ScoredChunk(
                    chunk=candidate.chunk,
                    score=float(score),
                    source="reranker",
                    rank=rank,
                )
            )
        return reranked

    @staticmethod
    @lru_cache(maxsize=2)
    def _load_model(model_name: str) -> CrossEncoder:
        logger.info("Loading cross-encoder reranker", extra={"ctx_model_name": model_name})
        return CrossEncoder(model_name)
