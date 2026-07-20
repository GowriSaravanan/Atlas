# Evaluation Report

Generated: 2026-07-20T18:49:32.433432+00:00

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
- rerank_lift_measured: **yes**
- answer_generation_measured: **yes**

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
- MRR: 0.7293
- nDCG@k: 0.7968

## Rerank

- Pre-rerank Recall@k: 1.0
- Post-rerank Recall@k: 1.0
- Recall delta: 0.0
- Pre-rerank MRR: 0.7293
- Post-rerank MRR: 0.7293
- MRR delta: 0.0
- Avg rerank latency: 0.01ms

## Answer Generation

- Generation success rate: 1.0
- Groundedness rate: 0.4
- Avg latency: 0.13ms
- Avg prompt tokens: 200.7
- Avg completion tokens: 26.9

## Rewrite

- Exact match rate: 1.0

## Routing

- Router accuracy: 1.0

## Decomposition

- Precision: 1.0
- Recall: 1.0
- False decomposition rate: 0.0

## Latency

- analysis: avg=0.13ms p95=0.14ms p99=2.53ms
- answer_generation: avg=0.2ms p95=0.35ms p99=4.2ms
- confidence: avg=0.0ms p95=0.0ms p99=0.0ms
- decomposition: avg=0.0ms p95=0.0ms p99=0.0ms
- merge: avg=0.0ms p95=0.0ms p99=0.0ms
- rerank: avg=0.01ms p95=0.03ms p99=0.04ms
- retrieval: avg=0.26ms p95=0.7ms p99=3.58ms
- rewrite: avg=0.0ms p95=0.0ms p99=0.0ms
