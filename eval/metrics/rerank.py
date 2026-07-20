"""Reranker evaluation metrics."""

from __future__ import annotations

from typing import Any

from eval.metrics.retrieval import mrr, recall_at_k, resolve_gold_ids


def evaluate_rerank_case(
    *,
    case_id: str,
    pre_rerank_ids: list[str],
    post_rerank_ids: list[str],
    catalog: list[tuple[str, str, dict[str, Any]]],
    gold_specs: list[dict[str, Any]],
    top_k: int,
    rerank_ms: float,
) -> dict[str, Any]:
    """Compare retrieval quality before and after reranking for one case."""
    gold_set = resolve_gold_ids(catalog, gold_specs)
    pre_recall = recall_at_k(pre_rerank_ids, gold_set, top_k)
    post_recall = recall_at_k(post_rerank_ids, gold_set, top_k)
    pre_mrr = mrr(pre_rerank_ids, gold_set)
    post_mrr = mrr(post_rerank_ids, gold_set)

    return {
        "id": case_id,
        "pre_recall_at_k": pre_recall,
        "post_recall_at_k": post_recall,
        "recall_delta": round(post_recall - pre_recall, 4),
        "pre_mrr": pre_mrr,
        "post_mrr": post_mrr,
        "mrr_delta": round(post_mrr - pre_mrr, 4),
        "rank_changed": pre_rerank_ids[:top_k] != post_rerank_ids[:top_k],
        "rerank_ms": round(rerank_ms, 2),
        "gold_count": len(gold_set),
    }


def summarize_rerank(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate before/after rerank metrics."""
    if not cases:
        return {"count": 0}

    count = len(cases)
    return {
        "count": count,
        "pre_recall_at_k": round(sum(case["pre_recall_at_k"] for case in cases) / count, 4),
        "post_recall_at_k": round(sum(case["post_recall_at_k"] for case in cases) / count, 4),
        "recall_delta": round(sum(case["recall_delta"] for case in cases) / count, 4),
        "pre_mrr": round(sum(case["pre_mrr"] for case in cases) / count, 4),
        "post_mrr": round(sum(case["post_mrr"] for case in cases) / count, 4),
        "mrr_delta": round(sum(case["mrr_delta"] for case in cases) / count, 4),
        "rank_change_rate": round(sum(1 for case in cases if case["rank_changed"]) / count, 4),
        "avg_rerank_ms": round(sum(case["rerank_ms"] for case in cases) / count, 2),
        "p95_rerank_ms": round(_percentile([case["rerank_ms"] for case in cases], 95), 2),
    }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[index]
