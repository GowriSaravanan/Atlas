"""Context builder unit tests."""

from __future__ import annotations

from adaptive_rag.config.settings import AnswerGenerationSettings
from adaptive_rag.domain.models.document import Chunk
from adaptive_rag.domain.models.retrieval import ScoredChunk
from adaptive_rag.domain.policies.context_builder import ContextBuilder


def _chunk(chunk_id: str, content: str, **metadata: str) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(id=chunk_id, document_id="doc-1", content=content, metadata=dict(metadata)),
        score=0.9,
        source="reranker",
        rank=1,
    )


def test_context_builder_orders_by_rank_and_preserves_metadata() -> None:
    builder = ContextBuilder(AnswerGenerationSettings(max_context_tokens=512))
    evidence = [
        ScoredChunk(
            chunk=Chunk(id="b", document_id="doc-1", content="second chunk", metadata={"section_title": "B"}),
            score=0.8,
            source="reranker",
            rank=2,
        ),
        ScoredChunk(
            chunk=Chunk(id="a", document_id="doc-1", content="first chunk", metadata={"section_title": "A"}),
            score=0.9,
            source="reranker",
            rank=1,
        ),
    ]

    built = builder.build(evidence)

    assert built.used_chunk_ids == ["a", "b"]
    assert "chunk_id=a" in built.context
    assert "section=A" in built.context
    assert built.context.index("first chunk") < built.context.index("second chunk")


def test_context_builder_enforces_token_budget() -> None:
    builder = ContextBuilder(AnswerGenerationSettings(max_context_tokens=256))
    evidence = [
        _chunk("c1", "word " * 200, section_title="One"),
        _chunk("c2", "word " * 200, section_title="Two"),
    ]

    built = builder.build(evidence)

    assert len(built.used_chunks) == 1
    assert built.truncated is True
    assert built.estimated_tokens <= 256


def test_context_builder_returns_empty_context_for_no_evidence() -> None:
    builder = ContextBuilder(AnswerGenerationSettings())
    built = builder.build([])

    assert built.context == ""
    assert built.used_chunk_ids == []
