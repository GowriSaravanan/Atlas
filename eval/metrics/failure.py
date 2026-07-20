"""Failure-mode evaluation metrics."""

from __future__ import annotations


def evaluate_failure_case(
    *,
    case_id: str,
    expected_behavior: str,
    result_count: int,
    confidence_value: float,
    confidence_acceptable: bool,
    was_rewritten: bool,
    query_type: str,
) -> dict[str, object]:
    """Evaluate graceful failure behavior."""
    if expected_behavior == "no_evidence":
        passed = result_count == 0 or (not confidence_acceptable and confidence_value < 0.65)
    elif expected_behavior == "low_confidence":
        passed = not confidence_acceptable or confidence_value < 0.65
    elif expected_behavior == "rewrite_or_clarify":
        passed = was_rewritten or query_type in {"ambiguous", "conversational"}
    elif expected_behavior == "no_false_decomposition":
        passed = True
    else:
        passed = not confidence_acceptable

    return {
        "id": case_id,
        "expected_behavior": expected_behavior,
        "passed": passed,
        "result_count": result_count,
        "confidence_value": round(confidence_value, 4),
        "confidence_acceptable": confidence_acceptable,
        "was_rewritten": was_rewritten,
        "query_type": query_type,
    }


def summarize_failure(cases: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate failure-mode pass rate."""
    if not cases:
        return {"count": 0}
    passed = sum(1 for case in cases if case["passed"])
    return {
        "count": len(cases),
        "pass_rate": round(passed / len(cases), 4),
        "passed": passed,
    }
