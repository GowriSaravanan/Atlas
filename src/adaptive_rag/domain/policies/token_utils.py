"""Token estimation utilities for chunking."""

from __future__ import annotations


def estimate_token_count(text: str) -> int:
    """Estimate token count using a simple words + punctuation heuristic."""
    if not text.strip():
        return 0
    words = text.split()
    return max(1, int(len(words) * 1.3))
