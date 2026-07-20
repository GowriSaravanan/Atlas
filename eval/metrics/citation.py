"""Citation formatting evaluation metrics."""

from __future__ import annotations

from typing import Any


def evaluate_citation_case(
    *,
    case_id: str,
    used_chunk_ids: list[str],
    citation_chunk_ids: list[str],
    min_citations: int = 1,
) -> dict[str, Any]:
    """Evaluate one citation-formatting benchmark case."""
    used_set = set(used_chunk_ids)
    cited_set = set(citation_chunk_ids)

    covered = used_set & cited_set
    coverage = len(covered) / len(used_set) if used_set else 1.0

    valid_citations = [chunk_id for chunk_id in citation_chunk_ids if chunk_id in used_set]
    precision = len(valid_citations) / len(citation_chunk_ids) if citation_chunk_ids else 1.0

    missing = [chunk_id for chunk_id in used_chunk_ids if chunk_id not in cited_set]
    missing_rate = len(missing) / len(used_set) if used_set else 0.0

    invalid = [chunk_id for chunk_id in citation_chunk_ids if chunk_id not in used_set]
    invalid_rate = len(invalid) / len(citation_chunk_ids) if citation_chunk_ids else 0.0

    expected_order = [chunk_id for chunk_id in used_chunk_ids if chunk_id in cited_set]
    order_preserved = citation_chunk_ids == expected_order

    return {
        "id": case_id,
        "citation_count": len(citation_chunk_ids),
        "min_citations_met": len(citation_chunk_ids) >= min_citations,
        "coverage": round(coverage, 4),
        "precision": round(precision, 4),
        "missing_citation_rate": round(missing_rate, 4),
        "invalid_citation_rate": round(invalid_rate, 4),
        "order_preserved": order_preserved,
        "missing_chunk_ids": missing,
        "invalid_chunk_ids": invalid,
    }


def summarize_citation(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate citation-formatting metrics."""
    if not cases:
        return {"count": 0}

    count = len(cases)
    return {
        "count": count,
        "citation_coverage": round(sum(case["coverage"] for case in cases) / count, 4),
        "citation_precision": round(sum(case["precision"] for case in cases) / count, 4),
        "missing_citation_rate": round(
            sum(case["missing_citation_rate"] for case in cases) / count,
            4,
        ),
        "invalid_citation_rate": round(
            sum(case["invalid_citation_rate"] for case in cases) / count,
            4,
        ),
        "order_preserved_rate": round(
            sum(1 for case in cases if case["order_preserved"]) / count,
            4,
        ),
        "min_citations_met_rate": round(
            sum(1 for case in cases if case["min_citations_met"]) / count,
            4,
        ),
    }
