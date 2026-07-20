"""Evidence citation formatting policy."""

from __future__ import annotations

import json
from typing import Any

from adaptive_rag.domain.config.policy_config import CitationFormatterPolicyConfig
from adaptive_rag.domain.models.answer import GeneratedAnswer
from adaptive_rag.domain.models.citation import Citation, CitationFormats
from adaptive_rag.domain.models.retrieval import ScoredChunk
from adaptive_rag.domain.ports.citation_formatter import CitationFormatterPort


def resolve_page_number(metadata: dict[str, Any]) -> int | None:
    """Resolve a single page number from chunk metadata."""
    for key in ("page_number", "page_start", "page_end"):
        value = metadata.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


class EvidenceCitationFormatter(CitationFormatterPort):
    """Build structured citations from reranked evidence used in answer generation."""

    def __init__(self, config: CitationFormatterPolicyConfig) -> None:
        self._config = config

    def format(
        self,
        answer: GeneratedAnswer,
        evidence: list[ScoredChunk],
    ) -> GeneratedAnswer:
        evidence_by_id = {hit.chunk.id: hit for hit in evidence}
        citations = self._build_citations(answer.used_chunk_ids, evidence_by_id)
        formats = self._build_formats(answer.answer, citations)
        return answer.model_copy(
            update={
                "citations": citations,
                "citation_formats": formats,
            }
        )

    def _build_citations(
        self,
        used_chunk_ids: list[str],
        evidence_by_id: dict[str, ScoredChunk],
    ) -> list[Citation]:
        citations: list[Citation] = []
        for chunk_id in used_chunk_ids:
            hit = evidence_by_id.get(chunk_id)
            if hit is None:
                continue
            chunk = hit.chunk
            metadata = chunk.metadata
            excerpt = chunk.content.strip()
            if len(excerpt) > self._config.excerpt_max_chars:
                excerpt = f"{excerpt[: self._config.excerpt_max_chars].rstrip()}..."
            citations.append(
                Citation(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    page_number=resolve_page_number(metadata),
                    section_title=str(metadata.get("section_title") or ""),
                    confidence=round(hit.score, 4) if hit.score is not None else None,
                    excerpt=excerpt,
                )
            )
        return citations

    def _build_formats(self, raw_answer: str, citations: list[Citation]) -> CitationFormats:
        return CitationFormats(
            markdown=self._format_markdown(raw_answer, citations),
            plain_text=self._format_plain_text(raw_answer, citations),
            json_text=self._format_json(raw_answer, citations),
        )

    def _format_markdown(self, answer: str, citations: list[Citation]) -> str:
        if not citations:
            return answer

        inline_refs = ", ".join(f"[{index}]" for index in range(1, len(citations) + 1))
        lines = [answer, "", f"**Sources:** {inline_refs}", "", "## References"]
        for index, citation in enumerate(citations, start=1):
            page_suffix = f", p.{citation.page_number}" if citation.page_number is not None else ""
            section = citation.section_title or "Untitled section"
            confidence_suffix = (
                f" (confidence: {citation.confidence:.2f})"
                if citation.confidence is not None
                else ""
            )
            lines.append(
                f"{index}. **{section}** — `{citation.document_id}`{page_suffix} "
                f"(`{citation.chunk_id}`){confidence_suffix}"
            )
            if citation.excerpt:
                lines.append(f"   > {citation.excerpt}")
        return "\n".join(lines)

    def _format_plain_text(self, answer: str, citations: list[Citation]) -> str:
        if not citations:
            return answer

        lines = [answer, "", "References:"]
        for index, citation in enumerate(citations, start=1):
            page_suffix = f", page {citation.page_number}" if citation.page_number is not None else ""
            section = citation.section_title or "Untitled section"
            confidence_suffix = (
                f" (confidence: {citation.confidence:.2f})"
                if citation.confidence is not None
                else ""
            )
            lines.append(
                f"[{index}] {section} ({citation.document_id}{page_suffix}) "
                f"[{citation.chunk_id}]{confidence_suffix}"
            )
            if citation.excerpt:
                lines.append(f"    {citation.excerpt}")
        return "\n".join(lines)

    def _format_json(self, answer: str, citations: list[Citation]) -> str:
        payload = {
            "answer": answer,
            "citations": [citation.model_dump(mode="json") for citation in citations],
        }
        return json.dumps(payload, indent=2, sort_keys=True)
