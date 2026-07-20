"""Answer generation evaluation metrics."""

from __future__ import annotations

from typing import Any


def evaluate_answer_case(
    *,
    case_id: str,
    answer: str,
    used_chunk_ids: list[str],
    expected_terms: list[str],
    min_chunks_used: int,
    latency_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, Any]:
    """Evaluate one answer-generation benchmark case."""
    normalized = answer.lower()
    grounded = all(term.lower() in normalized for term in expected_terms) if expected_terms else bool(
        answer.strip()
    )
    return {
        "id": case_id,
        "generated": bool(answer.strip()),
        "grounded": grounded,
        "chunks_used": len(used_chunk_ids),
        "min_chunks_met": len(used_chunk_ids) >= min_chunks_used,
        "latency_ms": round(latency_ms, 2),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "expected_terms": expected_terms,
    }


def summarize_answer(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate answer-generation metrics."""
    if not cases:
        return {"count": 0}

    count = len(cases)
    return {
        "count": count,
        "generation_success_rate": round(sum(1 for case in cases if case["generated"]) / count, 4),
        "groundedness_rate": round(sum(1 for case in cases if case["grounded"]) / count, 4),
        "min_chunks_met_rate": round(sum(1 for case in cases if case["min_chunks_met"]) / count, 4),
        "avg_latency_ms": round(sum(case["latency_ms"] for case in cases) / count, 2),
        "avg_prompt_tokens": round(sum(case["prompt_tokens"] for case in cases) / count, 2),
        "avg_completion_tokens": round(sum(case["completion_tokens"] for case in cases) / count, 2),
    }
