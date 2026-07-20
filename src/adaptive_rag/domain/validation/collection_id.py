"""Collection identifier validation and safe path resolution."""

from __future__ import annotations

import re
from pathlib import Path

from adaptive_rag.domain.errors import InvalidCollectionIdError

_COLLECTION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


def validate_collection_id(collection_id: str) -> str:
    """Validate collection_id against the allowed character set."""
    if not _COLLECTION_ID_PATTERN.fullmatch(collection_id):
        raise InvalidCollectionIdError(
            f"collection_id must match [a-zA-Z0-9_-]+; received {collection_id!r}"
        )
    return collection_id


def resolve_collection_path(base_path: Path, collection_id: str) -> Path:
    """Return a canonical collection path guaranteed to stay under base_path."""
    validate_collection_id(collection_id)
    root = base_path.resolve()
    collection_path = (root / collection_id).resolve()
    if root != collection_path and root not in collection_path.parents:
        raise InvalidCollectionIdError(
            f"collection_id resolves outside index directory: {collection_id!r}"
        )
    return collection_path
