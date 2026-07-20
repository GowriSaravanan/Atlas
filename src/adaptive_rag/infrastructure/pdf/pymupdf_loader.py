"""PyMuPDF-based intelligent PDF loader."""

from __future__ import annotations

import statistics
import uuid
from pathlib import Path
from typing import Any

import fitz

from adaptive_rag.domain.models.document import Document
from adaptive_rag.domain.ports.document_loader import DocumentLoaderPort
from adaptive_rag.observability.logging import get_logger

logger = get_logger(__name__)


class PyMuPDFLoader(DocumentLoaderPort):
    """Layout-aware PDF loader using heading detection and section grouping."""

    def load(self, source_path: str) -> Document:
        """Load a PDF and extract structured text with section metadata."""
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {source_path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"PyMuPDFLoader only supports PDF files: {source_path}")

        with fitz.open(source_path) as pdf:
            pages: list[str] = []
            sections: list[dict[str, Any]] = []
            current_section: dict[str, Any] | None = None

            for page_number, page in enumerate(pdf, start=1):
                blocks = self._extract_blocks(page)
                page_text = "\n".join(block["text"] for block in blocks if block["text"]).strip()
                pages.append(page_text)

                for block in blocks:
                    if not block["text"]:
                        continue

                    if block["is_heading"]:
                        if current_section and current_section["content"].strip():
                            sections.append(current_section)

                        current_section = {
                            "title": block["text"],
                            "content": "",
                            "page_start": page_number,
                            "page_end": page_number,
                        }
                        continue

                    if current_section is None:
                        current_section = {
                            "title": f"Page {page_number}",
                            "content": "",
                            "page_start": page_number,
                            "page_end": page_number,
                        }

                    current_section["content"] = (
                        f"{current_section['content']}\n{block['text']}".strip()
                    )
                    current_section["page_end"] = page_number

            if current_section and current_section["content"].strip():
                sections.append(current_section)

            full_text = "\n\n".join(page for page in pages if page).strip()
            if not full_text:
                raise ValueError(f"No extractable text found in PDF: {source_path}")

            metadata = {
                "source_type": "pdf",
                "file_name": path.name,
                "page_count": len(pdf),
                "title": sections[0]["title"] if sections else path.stem,
                "sections": sections,
            }

            logger.info(
                "Loaded PDF document",
                extra={
                    "ctx_source_path": source_path,
                    "ctx_page_count": len(pdf),
                    "ctx_section_count": len(sections),
                },
            )

            return Document(
                id=str(uuid.uuid4()),
                source_path=str(path),
                content=full_text,
                metadata=metadata,
            )

    @staticmethod
    def _extract_blocks(page: fitz.Page) -> list[dict[str, Any]]:
        raw_blocks = page.get_text("dict").get("blocks", [])
        font_sizes: list[float] = []

        parsed: list[dict[str, Any]] = []
        for block in raw_blocks:
            if block.get("type") != 0:
                continue

            lines = block.get("lines", [])
            text_parts: list[str] = []
            block_sizes: list[float] = []

            for line in lines:
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if text:
                        text_parts.append(text)
                        block_sizes.append(float(span.get("size", 12.0)))

            text = " ".join(text_parts).strip()
            if not text:
                continue

            avg_size = statistics.mean(block_sizes) if block_sizes else 12.0
            font_sizes.append(avg_size)
            parsed.append({"text": text, "font_size": avg_size, "is_heading": False})

        if not parsed:
            return parsed

        body_size = statistics.median(font_sizes)
        heading_threshold = body_size * 1.15

        for block in parsed:
            block["is_heading"] = (
                block["font_size"] >= heading_threshold and len(block["text"].split()) <= 16
            )

        return parsed
