"""Adaptive document chunking policy."""

from __future__ import annotations

import uuid
from typing import Any

from adaptive_rag.config.settings import ChunkingSettings
from adaptive_rag.domain.models.document import Chunk, Document
from adaptive_rag.domain.policies.token_utils import estimate_token_count
from adaptive_rag.domain.ports.chunker import ChunkerPort


class AdaptiveChunker(ChunkerPort):
    """Structure-aware chunking with fixed-size token fallback."""

    def __init__(self, settings: ChunkingSettings) -> None:
        self._settings = settings

    def chunk(self, document: Document) -> list[Chunk]:
        """Chunk a document using structure-aware or fixed strategy."""
        if self._settings.strategy == "fixed":
            return self._fixed_chunks(document)

        sections = document.metadata.get("sections")
        if isinstance(sections, list) and sections:
            chunks = self._chunk_by_sections(document, sections)
            if chunks:
                return chunks

        return self._fixed_chunks(document)

    def _chunk_by_sections(self, document: Document, sections: list[Any]) -> list[Chunk]:
        chunks: list[Chunk] = []
        index = 0

        for section in sections:
            if not isinstance(section, dict):
                continue

            title = str(section.get("title", "")).strip()
            content = str(section.get("content", "")).strip()
            if not content and not title:
                continue

            section_text = f"{title}\n\n{content}".strip() if title else content
            section_meta = {
                "section_title": title or None,
                "page_start": section.get("page_start"),
                "page_end": section.get("page_end"),
            }

            if estimate_token_count(section_text) <= self._settings.max_tokens:
                chunks.append(
                    self._build_chunk(
                        document=document,
                        content=section_text,
                        chunk_index=index,
                        extra_metadata=section_meta,
                    )
                )
                index += 1
                continue

            for piece in self._split_with_overlap(section_text):
                chunks.append(
                    self._build_chunk(
                        document=document,
                        content=piece,
                        chunk_index=index,
                        extra_metadata=section_meta,
                    )
                )
                index += 1

        return chunks

    def _fixed_chunks(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        for index, piece in enumerate(self._split_with_overlap(document.content)):
            chunks.append(
                self._build_chunk(
                    document=document,
                    content=piece,
                    chunk_index=index,
                    extra_metadata={},
                )
            )
        return chunks

    def _split_with_overlap(self, text: str) -> list[str]:
        words = text.split()
        if not words:
            return []

        max_words = max(1, int(self._settings.max_tokens / 1.3))
        overlap_words = max(0, int(self._settings.overlap_tokens / 1.3))
        step = max(1, max_words - overlap_words)

        pieces: list[str] = []
        start = 0
        while start < len(words):
            end = min(len(words), start + max_words)
            piece = " ".join(words[start:end]).strip()
            if piece and estimate_token_count(piece) >= self._settings.min_tokens or end == len(words):
                pieces.append(piece)
            if end >= len(words):
                break
            start += step

        return pieces or [text.strip()]

    def _build_chunk(
        self,
        *,
        document: Document,
        content: str,
        chunk_index: int,
        extra_metadata: dict[str, Any],
    ) -> Chunk:
        metadata = {
            **document.metadata,
            **{key: value for key, value in extra_metadata.items() if value is not None},
            "chunk_strategy": self._settings.strategy,
        }
        return Chunk(
            id=str(uuid.uuid4()),
            document_id=document.id,
            content=content,
            metadata=metadata,
            token_count=estimate_token_count(content),
            chunk_index=chunk_index,
        )
