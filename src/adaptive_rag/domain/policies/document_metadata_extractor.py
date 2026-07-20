"""Document metadata extraction policy."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from adaptive_rag.domain.models.document import Chunk, Document
from adaptive_rag.domain.ports.metadata_extractor import MetadataExtractorPort

_POLICY_ID_PATTERN = re.compile(r"\b([A-Z]{2,10}-\d{2,6})\b")


class DocumentMetadataExtractor(MetadataExtractorPort):
    """Rule-based metadata extraction for documents and chunks."""

    _DEPARTMENT_PATTERN = re.compile(
        r"\b(HR|Finance|Legal|Engineering|Operations|IT|Sales|Marketing)\b",
        re.IGNORECASE,
    )
    _DATE_PATTERN = re.compile(r"\b(20\d{2}[-/]\d{2}[-/]\d{2}|Q[1-4]\s+20\d{2})\b")

    def extract_document_metadata(self, document: Document) -> dict[str, Any]:
        """Extract and enrich document-level metadata."""
        path = Path(document.source_path)
        content_preview = document.content[:2000]

        departments = sorted({match.group(0).upper() for match in self._DEPARTMENT_PATTERN.finditer(content_preview)})
        dates = [match.group(0) for match in self._DATE_PATTERN.finditer(content_preview)]
        policy_ids = _POLICY_ID_PATTERN.findall(content_preview)

        metadata = {
            **document.metadata,
            "file_name": path.name,
            "file_extension": path.suffix.lower(),
            "source_type": document.metadata.get("source_type", path.suffix.lstrip(".") or "unknown"),
            "char_count": len(document.content),
            "detected_departments": departments,
            "detected_dates": dates[:5],
            "detected_policy_ids": policy_ids[:10],
        }

        if departments:
            metadata["department"] = departments[0]

        if "title" not in metadata or not metadata["title"]:
            metadata["title"] = self._infer_title(document)

        return metadata

    def extract_chunk_metadata(self, chunk: Chunk) -> dict[str, Any]:
        """Enrich chunk metadata for scoped retrieval."""
        metadata = dict(chunk.metadata)
        metadata.setdefault("document_id", chunk.document_id)
        metadata.setdefault("chunk_index", chunk.chunk_index)
        metadata.setdefault("token_count", chunk.token_count)

        policy_ids = _POLICY_ID_PATTERN.findall(chunk.content)
        if policy_ids:
            metadata["policy_id"] = policy_ids[0]
            metadata["policy_ids"] = policy_ids

        return metadata

    def extract_query_metadata(self, analysis: Any) -> dict[str, Any]:
        """Derive search filters from query analysis (Phase 3+)."""
        _ = analysis
        return {}

    def apply_to_document(self, document: Document) -> Document:
        """Return a copy of the document with enriched metadata."""
        enriched = self.extract_document_metadata(document)
        return document.model_copy(update={"metadata": enriched})

    def apply_to_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Return chunks with enriched metadata."""
        return [chunk.model_copy(update={"metadata": self.extract_chunk_metadata(chunk)}) for chunk in chunks]

    @staticmethod
    def _infer_title(document: Document) -> str:
        sections = document.metadata.get("sections")
        if isinstance(sections, list) and sections:
            first = sections[0]
            if isinstance(first, dict):
                title = str(first.get("title", "")).strip()
                if title:
                    return title

        for line in document.content.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped[:120]

        return Path(document.source_path).stem
