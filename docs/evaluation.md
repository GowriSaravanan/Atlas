# Evaluation Framework

The Adaptive Retrieval Platform is frozen at the retrieval layer until benchmarks pass. Do not start Phase 5 (answer generation) until the Phase 5 readiness checklist is green.

## Layout

```
eval/
  datasets/
    retrieval_dataset.jsonl
    rewrite_dataset.jsonl
    routing_dataset.jsonl
    decomposition_dataset.jsonl
    confidence_dataset.jsonl
    failure_dataset.jsonl
    answer_generation_dataset.jsonl
    citation_dataset.jsonl
    golden_demo.jsonl
  fixtures/
    corpus.py          # deterministic eval PDF
    catalog.py         # gold chunk resolution
  metrics/
    retrieval.py       # Recall@k, MRR, nDCG
    rewrite.py         # exact match
    routing.py         # router accuracy
    decomposition.py   # precision, recall, false decomposition rate
    confidence.py      # bucket calibration
    latency.py         # avg / p95 / p99 per stage
    failure.py         # graceful failure behavior
  reports/
    latest.json
    latest.md
  run_eval.py
```

## Run

```bash
uv run python eval/run_eval.py
uv run python eval/run_eval.py --suite decomposition
uv run python eval/run_eval.py --report eval/reports/latest.json
```

## Metrics

| Suite | Metrics |
|---|---|
| Retrieval | Recall@k, MRR, nDCG@k |
| Rewrite | Exact match (no BLEU) |
| Routing | Router accuracy |
| Decomposition | Precision, recall, false decomposition rate, subquery exact match |
| Confidence | Bucket match vs high/medium/low |
| Failure | Pass rate for no-evidence / low-confidence / clarify cases |
| Latency | avg, p95, p99 per pipeline stage |
| Rerank | Pre/post Recall@k, MRR delta, rerank latency |
| Answer generation | Generation success, groundedness, latency, tokens |
| Citation | Coverage, precision, missing/invalid citation rate, order preserved |
| Golden demo | 10 interview-ready traces |

## Phase 5 Readiness Checklist

The report includes:

- rewrite accuracy measured
- router accuracy measured
- decomposition accuracy measured
- Recall@k measured
- MRR measured
- false decomposition rate measured
- latency per stage measured
- failure cases evaluated
- confidence evaluated
- citation formatting evaluated

All must be **yes** (instrumentation complete) before treating benchmarks as operational.

Quality gates in the report must pass before Phase 5:

| Gate | Threshold |
|---|---|
| `router_accuracy_gte_0_85` | ≥85% |
| `recall_at_k_gte_0_75` | ≥75% |
| `mrr_gte_0_60` | ≥60% |
| `rewrite_exact_match_rate_gte_0_75` | ≥75% |
| `false_decomposition_rate_lte_0_10` | ≤10% |
| `failure_pass_rate_gte_0_75` | ≥75% |

See [improvement-log.md](../eval/reports/improvement-log.md) for before/after metrics.

## Gold Labels

Retrieval datasets use portable gold specs instead of brittle chunk UUIDs:

```json
{"gold": [{"policy_id": "HR-203"}]}
{"gold": [{"section_title": "Maternity Leave"}]}
{"gold": [{"content_contains": "26 weeks"}]}
```

Gold ids are resolved against the ingested eval corpus catalog at runtime.

## Golden Demo

Run the 10-query demo set after every major change:

```bash
uv run python eval/run_eval.py --suite golden --report eval/reports/golden_demo.json
```

Use this in interviews to show routing, rewrite, decomposition, confidence, and latency traces end-to-end.
