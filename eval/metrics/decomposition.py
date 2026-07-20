"""Decomposition evaluation metrics."""

from __future__ import annotations


def _normalize_query(text: str) -> str:
    return text.strip().rstrip("?").lower()


def evaluate_decomposition_case(
    *,
    case_id: str,
    should_decompose: bool,
    was_decomposed: bool,
    actual_subqueries: list[str],
    expected_subqueries: list[str] | None = None,
) -> dict[str, object]:
    """Evaluate one decomposition benchmark case."""
    expected = expected_subqueries or []
    normalized_actual = {_normalize_query(query) for query in actual_subqueries}
    normalized_expected = {_normalize_query(query) for query in expected}

    subquery_exact_match = normalized_actual == normalized_expected if should_decompose else True
    false_decomposition = (not should_decompose) and was_decomposed
    missed_decomposition = should_decompose and (not was_decomposed)

    return {
        "id": case_id,
        "should_decompose": should_decompose,
        "was_decomposed": was_decomposed,
        "false_decomposition": false_decomposition,
        "missed_decomposition": missed_decomposition,
        "decision_correct": should_decompose == was_decomposed,
        "subquery_exact_match": subquery_exact_match,
        "expected_subqueries": expected,
        "actual_subqueries": actual_subqueries,
    }


def summarize_decomposition(cases: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate decomposition precision, recall, and false decomposition rate."""
    if not cases:
        return {"count": 0}

    predicted_positive = [case for case in cases if case["was_decomposed"]]
    actual_positive = [case for case in cases if case["should_decompose"]]

    true_positive = sum(
        1 for case in cases if case["should_decompose"] and case["was_decomposed"]
    )
    false_positive = sum(1 for case in cases if case["false_decomposition"])
    false_negative = sum(1 for case in cases if case["missed_decomposition"])

    precision = true_positive / len(predicted_positive) if predicted_positive else 1.0
    recall = true_positive / len(actual_positive) if actual_positive else 1.0
    false_decomposition_rate = false_positive / len(cases)
    subquery_match_rate = sum(1 for case in cases if case["subquery_exact_match"]) / len(cases)

    return {
        "count": len(cases),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "false_decomposition_rate": round(false_decomposition_rate, 4),
        "subquery_exact_match_rate": round(subquery_match_rate, 4),
        "decision_accuracy": round(sum(1 for case in cases if case["decision_correct"]) / len(cases), 4),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }
