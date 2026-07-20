"""Routing evaluation metrics."""

from __future__ import annotations


def evaluate_routing_case(
    *,
    case_id: str,
    expected_strategy: str,
    actual_strategy: str,
    reason: str,
) -> dict[str, object]:
    """Evaluate one routing benchmark case."""
    return {
        "id": case_id,
        "expected_strategy": expected_strategy,
        "actual_strategy": actual_strategy,
        "correct": expected_strategy == actual_strategy,
        "reason": reason,
    }


def summarize_routing(cases: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate routing accuracy."""
    if not cases:
        return {"count": 0}
    correct = sum(1 for case in cases if case["correct"])
    return {
        "count": len(cases),
        "router_accuracy": round(correct / len(cases), 4),
        "correct": correct,
    }
