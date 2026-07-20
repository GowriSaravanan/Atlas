"""Latency evaluation metrics."""

from __future__ import annotations

from collections import defaultdict


STAGE_ALIASES = {
    "original_query_analysis": "analysis",
    "resolved_query_analysis": "analysis",
    "query_rewrite": "rewrite",
    "query_decomposition": "decomposition",
    "adaptive_routing": "routing",
    "subquery_merge": "merge",
    "rerank": "rerank",
    "answer_generation": "answer_generation",
    "confidence_scoring": "confidence",
    "dense_retrieval": "retrieval",
    "sparse_retrieval": "retrieval",
    "rrf_fusion": "retrieval",
}


def stage_for_step(step_name: str) -> str:
    """Map a trace step name to a latency stage bucket."""
    if step_name.startswith("subquery_retrieval_"):
        return "retrieval"
    return STAGE_ALIASES.get(step_name, "other")


def percentile(values: list[float], pct: float) -> float:
    """Compute a percentile from a list of values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[index]


def summarize_latency(traces: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate avg/p95/p99 latency per stage from trace payloads."""
    buckets: dict[str, list[float]] = defaultdict(list)

    for trace in traces:
        for step in trace.get("steps", []):
            stage = stage_for_step(str(step.get("step", "")))
            buckets[stage].append(float(step.get("duration_ms", 0.0)))

        for key, value in trace.get("latency_ms", {}).items():
            if key.endswith("_ms"):
                stage = "analysis" if "analyze" in key else "retrieval"
                buckets[stage].append(float(value))

    summary: dict[str, object] = {}
    for stage, values in sorted(buckets.items()):
        summary[stage] = {
            "avg_ms": round(sum(values) / len(values), 2),
            "p95_ms": round(percentile(values, 95), 2),
            "p99_ms": round(percentile(values, 99), 2),
            "samples": len(values),
        }
    return summary
