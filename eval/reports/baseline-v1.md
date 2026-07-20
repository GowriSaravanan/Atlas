# Evaluation Report

Generated: 2026-07-20T18:27:10.217615+00:00

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
**Ready for Phase 5:** no

## Quality Gates

- rewrite_exact_match_rate_gte_0_75: **pass**
- router_accuracy_gte_0_80: **fail**
- recall_at_k_gte_0_70: **fail**
- mrr_gte_0_50: **fail**
- false_decomposition_rate_lte_0_10: **pass**
- failure_pass_rate_gte_0_75: **pass**

## Retrieval

- Recall@k: 0.3333
- MRR: 0.3
- nDCG@k: 0.2996

## Rewrite

- Exact match rate: 1.0

## Routing

- Router accuracy: 0.2

## Decomposition

- Precision: 1.0
- Recall: 0.6667
- False decomposition rate: 0.0

## Latency

- analysis: avg=0.13ms p95=0.33ms p99=1.42ms
- confidence: avg=0.0ms p95=0.0ms p99=0.0ms
- decomposition: avg=0.0ms p95=0.0ms p99=0.0ms
- merge: avg=0.0ms p95=0.0ms p99=0.0ms
- retrieval: avg=0.71ms p95=1.49ms p99=11.4ms
- rewrite: avg=0.0ms p95=0.0ms p99=0.0ms
