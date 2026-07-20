"""Reranker adapter and pipeline integration tests."""

from __future__ import annotations

from adaptive_rag.application.services.retrieval_engine import RetrievalEngine
from adaptive_rag.domain.models.document import Chunk
from adaptive_rag.domain.models.retrieval import ScoredChunk
from adaptive_rag.infrastructure.reranking.fake_reranker import FakeReranker


class _RecordingReranker(FakeReranker):
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int,
    ) -> list[ScoredChunk]:
        self.calls.append((query, len(candidates), top_k))
        return super().rerank(query, candidates, top_k)


def _chunk(chunk_id: str, content: str) -> Chunk:
    return Chunk(id=chunk_id, document_id="doc-1", content=content, metadata={})


def test_fake_reranker_preserves_order_and_tags_source() -> None:
    reranker = FakeReranker()
    candidates = [
        ScoredChunk(chunk=_chunk("a", "alpha"), score=0.9, source="rrf", rank=1),
        ScoredChunk(chunk=_chunk("b", "beta"), score=0.8, source="rrf", rank=2),
    ]

    reranked = reranker.rerank("alpha query", candidates, top_k=2)

    assert [hit.chunk.id for hit in reranked] == ["a", "b"]
    assert all(hit.source == "reranker" for hit in reranked)
    assert [hit.rank for hit in reranked] == [1, 2]


def test_cross_encoder_reranker_reorders_by_score(monkeypatch) -> None:
    from adaptive_rag.config.settings import RerankerSettings
    from adaptive_rag.infrastructure.reranking.cross_encoder import CrossEncoderReranker

    class _FakeCrossEncoder:
        def predict(self, pairs, batch_size=16, show_progress_bar=False):
            del batch_size, show_progress_bar
            scores = []
            for query, content in pairs:
                scores.append(len(content) - len(query))
            return scores

    monkeypatch.setattr(
        CrossEncoderReranker,
        "_load_model",
        staticmethod(lambda _model_name: _FakeCrossEncoder()),
    )

    reranker = CrossEncoderReranker(RerankerSettings())
    candidates = [
        ScoredChunk(chunk=_chunk("short", "short text"), score=0.2, source="rrf"),
        ScoredChunk(chunk=_chunk("long", "much longer candidate text"), score=0.9, source="rrf"),
    ]

    reranked = reranker.rerank("q", candidates, top_k=2)

    assert [hit.chunk.id for hit in reranked] == ["long", "short"]
    assert reranked[0].source == "reranker"
    assert reranked[0].rank == 1


def test_retrieval_engine_reranks_before_confidence(sample_pdf) -> None:
    from adaptive_rag.api.dependencies.container import get_container
    from adaptive_rag.domain.policies.rrf import ReciprocalRankFusion

    from adaptive_rag.config.mappers import to_fusion_policy_config

    container = get_container()
    container.ensure_storage_dirs()
    container.ingest_document_use_case.execute(
        source_path=str(sample_pdf),
        collection_id="default",
    )

    recording_reranker = _RecordingReranker()
    engine = RetrievalEngine(
        index_registry=container.index_registry,
        hybrid_retriever=container.hybrid_retriever,
        settings=container.settings.retrieval,
        fusion_engine=ReciprocalRankFusion(to_fusion_policy_config(container.settings.retrieval)),
        confidence_scorer=container.confidence_scorer,
        reranker=recording_reranker,
    )

    result = engine.execute(query="How many sick leave days?", collection_id="default", top_k=5)

    assert recording_reranker.calls
    rerank_step = next(step for step in result.trace.steps if step.step == "rerank")
    assert rerank_step.metadata["skipped"] is False
    assert result.trace.reranked_hits
    assert all(hit.source == "reranker" for hit in result.results)
    confidence_index = next(
        index for index, step in enumerate(result.trace.steps) if step.step == "confidence_scoring"
    )
    rerank_index = next(index for index, step in enumerate(result.trace.steps) if step.step == "rerank")
    assert rerank_index < confidence_index
