"""Confidence evaluation metrics."""

from __future__ import annotations

CONFIDENCE_BUCKETS = {
    "high": (0.65, 1.01),
    "medium": (0.40, 0.65),
    "low": (-0.01, 0.40),
}


def bucket_for_value(value: float) -> str:
    """Map a confidence score to high/medium/low."""
    for name, (lower, upper) in CONFIDENCE_BUCKETS.items():
        if lower <= value < upper:
            return name
    return "low"


def evaluate_confidence_case(
    *,
    case_id: str,
    expected_confidence: str,
    actual_value: float,
    retrieval_success: bool,
) -> dict[str, object]:
    """Evaluate predicted confidence bucket against expectation."""
    predicted_bucket = bucket_for_value(actual_value)
    bucket_match = predicted_bucket == expected_confidence
    success_alignment = (
        (expected_confidence == "high" and retrieval_success)
        or (expected_confidence in {"medium", "low"} and not retrieval_success)
        or (expected_confidence == "medium" and retrieval_success)
        or (expected_confidence == "low" and not retrieval_success)
    )
    return {
        "id": case_id,
        "expected_confidence": expected_confidence,
        "predicted_bucket": predicted_bucket,
        "actual_value": round(actual_value, 4),
        "bucket_match": bucket_match,
        "retrieval_success": retrieval_success,
        "success_alignment": success_alignment,
    }


def summarize_confidence(cases: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate confidence calibration metrics."""
    if not cases:
        return {"count": 0}
    return {
        "count": len(cases),
        "bucket_match_rate": round(sum(1 for case in cases if case["bucket_match"]) / len(cases), 4),
        "success_alignment_rate": round(
            sum(1 for case in cases if case["success_alignment"]) / len(cases), 4
        ),
    }
