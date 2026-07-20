"""Filesystem path helpers for ingestion security."""

from __future__ import annotations

from pathlib import Path

from adaptive_rag.domain.errors import PathAccessError

PDF_MAGIC = b"%PDF"
DEFAULT_MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def resolve_path_within_directory(source_path: str, allowed_root: Path) -> Path:
    """Resolve source_path and ensure it remains inside allowed_root."""
    root = allowed_root.resolve()
    candidate = Path(source_path)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()

    if root != resolved and root not in resolved.parents:
        raise PathAccessError(f"Path must be inside {root}; received {source_path!r}")

    if not resolved.exists():
        raise FileNotFoundError(f"Document not found: {source_path}")

    return resolved


def validate_pdf_upload(content: bytes, *, max_bytes: int = DEFAULT_MAX_UPLOAD_BYTES) -> None:
    """Validate uploaded PDF content size and magic bytes."""
    if len(content) == 0:
        raise ValueError("Uploaded file is empty")
    if len(content) > max_bytes:
        raise ValueError(f"Uploaded file exceeds maximum size of {max_bytes} bytes")
    if not content.startswith(PDF_MAGIC):
        raise ValueError("Uploaded file is not a valid PDF")
