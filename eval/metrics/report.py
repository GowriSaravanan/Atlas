"""Evaluation report builder."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def _quality_gates(sections: dict[str, Any]) -> dict[str, bool]:
    """Return quality gate results used before starting Phase 5."""
    rewrite_rate = sections.get("rewrite", {}).get("summary", {}).get("exact_match_rate", 0.0)
    router_accuracy = sections.get("routing", {}).get("summary", {}).get("router_accuracy", 0.0)
    recall_at_k = sections.get("retrieval", {}).get("summary", {}).get("recall_at_k", 0.0)
    mrr = sections.get("retrieval", {}).get("summary", {}).get("mrr", 0.0)
    false_decomposition = sections.get("decomposition", {}).get("summary", {}).get(
        "false_decomposition_rate", 1.0
    )
    failure_pass = sections.get("failure", {}).get("summary", {}).get("pass_rate", 0.0)

    return {
        "rewrite_exact_match_rate_gte_0_75": rewrite_rate >= 0.75,
        "router_accuracy_gte_0_85": router_accuracy >= 0.85,
        "recall_at_k_gte_0_75": recall_at_k >= 0.75,
        "mrr_gte_0_60": mrr >= 0.60,
        "false_decomposition_rate_lte_0_10": false_decomposition <= 0.10,
        "failure_pass_rate_gte_0_75": failure_pass >= 0.75,
    }


def build_summary_report(sections: dict[str, Any]) -> dict[str, Any]:
    """Build a top-level evaluation report payload."""
    checklist = {
        "rewrite_accuracy_measured": "rewrite" in sections,
        "router_accuracy_measured": "routing" in sections,
        "decomposition_accuracy_measured": "decomposition" in sections,
        "recall_at_k_measured": sections.get("retrieval", {}).get("summary", {}).get("recall_at_k") is not None,
        "mrr_measured": sections.get("retrieval", {}).get("summary", {}).get("mrr") is not None,
        "false_decomposition_rate_measured": sections.get("decomposition", {}).get("summary", {}).get(
            "false_decomposition_rate"
        )
        is not None,
        "latency_per_stage_measured": bool(sections.get("latency")),
        "failure_cases_evaluated": "failure" in sections,
        "confidence_evaluated": "confidence" in sections,
    }
    quality_gates = _quality_gates(sections)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "instrumentation_complete": all(checklist.values()),
        "phase_5_readiness": checklist,
        "quality_gates": quality_gates,
        "ready_for_phase_5": all(quality_gates.values()),
        "sections": sections,
    }
