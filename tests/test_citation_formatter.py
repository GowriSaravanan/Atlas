"""Citation formatter unit and integration tests."""

from __future__ import annotations

import json

from adaptive_rag.domain.config.policy_config import CitationFormatterPolicyConfig
from adaptive_rag.domain.models.answer import GeneratedAnswer
from adaptive_rag.domain.models.document import Chunk
from adaptive_rag.domain.models.retrieval import ScoredChunk
from adaptive_rag.domain.policies.evidence_citation_formatter import (
    EvidenceCitationFormatter,
    resolve_page_number,
)
from adaptive_rag.infrastructure.reranking.fake_reranker import FakeReranker


def _chunk(
    chunk_id: str,
    *,
    document_id: str = "doc-1",
    content: str = "Employees may take up to 10 sick leave days per year.",
    metadata: dict | None = None,
    score: float = 0.91,
    rank: int = 1,
) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            id=chunk_id,
            document_id=document_id,
            content=content,
            metadata=metadata or {"section_title": "Sick Leave", "page_start": 2},
        ),
        score=score,
        source="reranker",
        rank=rank,
    )


def _answer(*, used_chunk_ids: list[str], answer: str = "Based on the evidence.") -> GeneratedAnswer:
    return GeneratedAnswer(
        answer=answer,
        used_chunk_ids=used_chunk_ids,
        model_name="fake-answer-generator",
    )


def test_resolve_page_number_prefers_page_start() -> None:
    assert resolve_page_number({"page_start": 3, "page_end": 4}) == 3
    assert resolve_page_number({"page_number": 5}) == 5
    assert resolve_page_number({}) is None


def test_formatter_builds_citations_in_used_chunk_order() -> None:
    formatter = EvidenceCitationFormatter(
        CitationFormatterPolicyConfig(excerpt_max_chars=200)
    )
    evidence = [
        _chunk("c1", rank=1, score=0.95),
        _chunk("c2", rank=2, score=0.80, metadata={"section_title": "Annual Leave", "page_start": 1}),
    ]
    result = formatter.format(_answer(used_chunk_ids=["c2", "c1"]), evidence)

    assert [citation.chunk_id for citation in result.citations] == ["c2", "c1"]
    assert result.citations[0].section_title == "Annual Leave"
    assert result.citations[0].page_number == 1
    assert result.citations[0].confidence == 0.8
    assert result.citations[1].confidence == 0.95
    assert result.answer == "Based on the evidence."


def test_formatter_skips_unknown_chunk_ids() -> None:
    formatter = EvidenceCitationFormatter(
        CitationFormatterPolicyConfig(excerpt_max_chars=200)
    )
    evidence = [_chunk("c1")]
    result = formatter.format(_answer(used_chunk_ids=["missing", "c1"]), evidence)

    assert [citation.chunk_id for citation in result.citations] == ["c1"]


def test_formatter_produces_markdown_plain_and_json_formats() -> None:
    formatter = EvidenceCitationFormatter(
        CitationFormatterPolicyConfig(excerpt_max_chars=80)
    )
    evidence = [_chunk("c1")]
    result = formatter.format(_answer(used_chunk_ids=["c1"], answer="Ten sick days."), evidence)

    assert result.citation_formats is not None
    assert "Ten sick days." in result.citation_formats.markdown
    assert "**Sources:** [1]" in result.citation_formats.markdown
    assert "References:" in result.citation_formats.plain_text
    assert "[1]" in result.citation_formats.plain_text

    payload = json.loads(result.citation_formats.json_text)
    assert payload["answer"] == "Ten sick days."
    assert payload["citations"][0]["chunk_id"] == "c1"
    assert payload["citations"][0]["section_title"] == "Sick Leave"


def test_formatter_truncates_excerpt() -> None:
    formatter = EvidenceCitationFormatter(
        CitationFormatterPolicyConfig(excerpt_max_chars=20)
    )
    long_content = "word " * 30
    evidence = [_chunk("c1", content=long_content.strip())]
    result = formatter.format(_answer(used_chunk_ids=["c1"]), evidence)

    assert result.citations[0].excerpt.endswith("...")
    assert len(result.citations[0].excerpt) <= 23


def test_retrieval_engine_runs_citation_formatting_after_answer_generation(sample_pdf) -> None:
    from adaptive_rag.api.dependencies.container import get_container

    container = get_container()
    container.ensure_storage_dirs()
    container.ingest_document_use_case.execute(source_path=str(sample_pdf), collection_id="default")

    result = container.hybrid_retrieval_use_case.execute(
        query="How many sick leave days?",
        collection_id="default",
        top_k=5,
    )

    generated = result.generated_answer
    assert generated is not None
    assert generated.citations
    assert generated.citation_formats is not None
    assert generated.citation_formats.markdown
    assert all(citation.chunk_id in generated.used_chunk_ids for citation in generated.citations)
    assert any(step.step == "citation_formatting" for step in result.trace.steps)


def test_retrieval_engine_preserves_rerank_order_in_citations() -> None:
    from adaptive_rag.domain.policies.evidence_citation_formatter import EvidenceCitationFormatter
    from adaptive_rag.infrastructure.reranking.fake_reranker import FakeReranker

    class _StubAnswerGenerator:
        def generate(self, query: str, evidence: list[ScoredChunk]):
            return GeneratedAnswer(
                answer="Answer",
                used_chunk_ids=[hit.chunk.id for hit in evidence],
                model_name="stub",
            )

    evidence = [
        _chunk("c-low", rank=2, score=0.4),
        _chunk("c-high", rank=1, score=0.9),
    ]
    reranker = FakeReranker()
    reranked = reranker.rerank("query", list(reversed(evidence)), top_k=2)
    answer = _StubAnswerGenerator().generate("query", reranked)
    formatted = EvidenceCitationFormatter(
        CitationFormatterPolicyConfig(excerpt_max_chars=200)
    ).format(answer, reranked)

    assert [citation.chunk_id for citation in formatted.citations] == [
        hit.chunk.id for hit in reranked
    ]
