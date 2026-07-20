"""Retrieval evaluation metrics."""

from __future__ import annotations

import math
from typing import Any


def chunk_matches_gold(chunk_content: str, chunk_metadata: dict[str, Any], gold: dict[str, Any]) -> bool:
    """Return True when a chunk satisfies a gold specification."""
    content = chunk_content.lower()
    if "content_contains" in gold:
        if gold["content_contains"].lower() not in content:
            return False
    if "section_title" in gold:
        section = chunk_metadata.get("section_title")
        target = gold["section_title"].lower()
        if section:
            if str(section).lower() != target:
                return False
        elif target not in content:
            return False
    if "policy_id" in gold:
        metadata_policy_id = chunk_metadata.get("policy_id")
        target = gold["policy_id"].upper()
        if metadata_policy_id:
            if str(metadata_policy_id).upper() != target:
                return False
        elif target not in chunk_content.upper():
            return False
    return True


def resolve_gold_ids(
    catalog: list[tuple[str, str, dict[str, Any]]],
    gold_specs: list[dict[str, Any]],
) -> set[str]:
    """Resolve gold chunk ids by scanning the ingested catalog."""
    gold_ids: set[str] = set()
    for chunk_id, content, metadata in catalog:
        if any(chunk_matches_gold(content, metadata, gold) for gold in gold_specs):
            gold_ids.add(chunk_id)
    return gold_ids


def recall_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    """Fraction of gold chunks found in the top-k retrieved set."""
    if not gold_ids:
        return 1.0
    top = set(retrieved_ids[:k])
    return len(top & gold_ids) / len(gold_ids)


def mrr(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    """Mean reciprocal rank for the first relevant hit."""
    if not gold_ids:
        return 1.0
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in gold_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    """Normalized discounted cumulative gain at k."""
    if not gold_ids:
        return 1.0

    dcg = 0.0
    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in gold_ids:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(gold_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def evaluate_retrieval_case(
    *,
    case_id: str,
    retrieved_ids: list[str],
    catalog: list[tuple[str, str, dict[str, Any]]],
    gold_specs: list[dict[str, Any]],
    expected_strategy: str | None,
    actual_strategy: str,
    top_k: int,
) -> dict[str, Any]:
    """Evaluate one retrieval benchmark case."""
    gold_set = resolve_gold_ids(catalog, gold_specs)
    return {
        "id": case_id,
        "recall_at_k": recall_at_k(retrieved_ids, gold_set, top_k),
        "mrr": mrr(retrieved_ids, gold_set),
        "ndcg_at_k": ndcg_at_k(retrieved_ids, gold_set, top_k),
        "strategy_match": expected_strategy is None or expected_strategy == actual_strategy,
        "expected_strategy": expected_strategy,
        "actual_strategy": actual_strategy,
        "gold_hits_in_results": len(set(retrieved_ids[:top_k]) & gold_set),
        "gold_count": len(gold_set),
    }


def summarize_retrieval(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate retrieval metrics."""
    if not cases:
        return {"count": 0}

    return {
        "count": len(cases),
        "recall_at_k": round(sum(case["recall_at_k"] for case in cases) / len(cases), 4),
        "mrr": round(sum(case["mrr"] for case in cases) / len(cases), 4),
        "ndcg_at_k": round(sum(case["ndcg_at_k"] for case in cases) / len(cases), 4),
        "router_accuracy": round(sum(1 for case in cases if case["strategy_match"]) / len(cases), 4),
    }
