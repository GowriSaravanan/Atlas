"""Rewrite evaluation metrics."""

from __future__ import annotations


def evaluate_rewrite_case(
    *,
    case_id: str,
    actual_rewrite: str,
    expected_rewrite: str,
    rewrite_type: str | None = None,
) -> dict[str, object]:
    """Evaluate rewrite exact match."""
    normalized_actual = actual_rewrite.strip().rstrip("?").lower()
    normalized_expected = expected_rewrite.strip().rstrip("?").lower()
    exact_match = normalized_actual == normalized_expected
    return {
        "id": case_id,
        "exact_match": exact_match,
        "rewrite_type": rewrite_type,
        "expected_rewrite": expected_rewrite,
        "actual_rewrite": actual_rewrite,
    }


def summarize_rewrite(cases: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate rewrite metrics."""
    if not cases:
        return {"count": 0}
    matches = sum(1 for case in cases if case["exact_match"])
    return {
        "count": len(cases),
        "exact_match_rate": round(matches / len(cases), 4),
        "exact_matches": matches,
    }
