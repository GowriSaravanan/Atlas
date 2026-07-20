# Evaluation Report

Generated: 2026-07-20T18:32:05.525680+00:00

## Phase 5 Readiness

- rewrite_accuracy_measured: **yes**
- router_accuracy_measured: **yes**
- decomposition_accuracy_measured: **yes**
- recall_at_k_measured: **yes**
- mrr_measured: **yes**
- false_decomposition_rate_measured: **yes**
- latency_per_stage_measured: **yes**
- failure_cases_evaluated: **yes**
- confidence_evaluated: **yes**

**Instrumentation complete:** yes
**Ready for Phase 5:** yes

## Quality Gates

- rewrite_exact_match_rate_gte_0_75: **pass**
- router_accuracy_gte_0_85: **pass**
- recall_at_k_gte_0_75: **pass**
- mrr_gte_0_60: **pass**
- false_decomposition_rate_lte_0_10: **pass**
- failure_pass_rate_gte_0_75: **pass**

## Retrieval

- Recall@k: 1.0
- MRR: 0.75
- nDCG@k: 0.8123

## Rewrite

- Exact match rate: 1.0

## Routing

- Router accuracy: 1.0

## Decomposition

- Precision: 1.0
- Recall: 1.0
- False decomposition rate: 0.0

## Latency

- analysis: avg=0.07ms p95=0.11ms p99=0.2ms
- confidence: avg=0.0ms p95=0.0ms p99=0.0ms
- decomposition: avg=0.0ms p95=0.0ms p99=0.0ms
- merge: avg=0.0ms p95=0.0ms p99=0.0ms
- retrieval: avg=0.33ms p95=0.97ms p99=1.63ms
- rewrite: avg=0.0ms p95=0.0ms p99=0.0ms
