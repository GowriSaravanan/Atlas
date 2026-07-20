# Phase 4B Design — Query Decomposition

**Status:** Implemented (Phase 4B complete)  
**Prerequisite:** Phase 4A (query rewriting) complete  
**Next:** Evaluation dataset (20–50 questions) before Phase 5 answer generation

---

## Goals

Split comparison and multi-intent queries into standalone subqueries, retrieve sequentially per subquery, and merge results — without disturbing the rewrite → analyze → route pipeline established in Phase 4A.

**Non-goals for 4B v1:**
- Parallel subquery execution (`asyncio.gather`)
- LLM-based decomposition (rules first, same pattern as analyzer/rewriter)
- Answer generation or reranking (Phase 5)

---

## Pipeline Position

```
Original Query
    ↓
Original Analysis          (OriginalQueryAnalysis)
    ↓
Rewrite? (Phase 4A)
    ↓
Resolved Query
    ↓
Resolved Analysis          (ResolvedQueryAnalysis)
    ↓
Decompose? (Phase 4B)      ← decision from resolved_analysis.decomposition_decision
    ↓
Subquery A → Retrieve → Subquery B → Retrieve → …   (sequential)
    ↓
Merge + dedupe
    ↓
Adaptive Router per subquery OR single strategy from resolved analysis
    ↓
Confidence (per subquery + aggregate)
```

Decomposition decisions are made on **resolved analysis**, not the original ambiguous query.

---

## Domain Models

### `SubQuery` (new — replaces bare `str`)

```python
class SubQuerySource(str, Enum):
    COMPARISON = "comparison"
    MULTI_HOP = "multi_hop"
    MULTI_QUESTION = "multi_question"

class SubQuery(BaseModel):
    id: str                          # "A", "B", …
    query: str                       # standalone retrieval text
    entity: str | None = None        # "maternity leave"
    source: SubQuerySource
    parent_query: str                # resolved query before decomposition
```

**Why not `list[str]`:** Phase 5 needs per-subquery confidence, citations, reranking, and partial answers. Structured subqueries make that tractable.

**Note:** `SubqueryState` / `SubquerySummary` already exist in `domain/models/retrieval.py` with `text: str`. Phase 4B will evolve these to reference `SubQuery` instead of duplicating fields.

### `DecompositionResult` (new)

```python
class DecompositionResult(BaseModel):
    original_query: str              # resolved query (post-rewrite)
    was_decomposed: bool
    reason: str
    subqueries: list[SubQuery] = Field(default_factory=list)
```

When `was_decomposed=False`, `subqueries` contains a single implicit subquery equal to the resolved query (or empty — TBD at implementation; prefer single-item list for uniform downstream handling).

### `QueryDecomposerPort` (new)

```python
class QueryDecomposerPort(Protocol):
    def decompose(
        self,
        query: str,
        analysis: ResolvedQueryAnalysis,
    ) -> DecompositionResult: ...
```

Implementations:
- `RuleBasedQueryDecomposer` (Phase 4B)
- `LLMQueryDecomposer` (future, via `LLMPort`)

---

## Decomposition Policy (Conservative)

Trigger only when `resolved_analysis.decomposition_decision.should_decompose` is true.

| Query | Decompose? | Reason |
|---|---|---|
| Compare maternity and paternity leave | Yes | distinct entities |
| Annual leave policy and carry over days | No | single policy topic |
| What is HR-203? | No | lookup |

Rule-based splitter for comparisons:
- Split on ` vs `, ` versus `, `compare X and Y`, `difference between X and Y`
- Normalize each arm into `"What is the {entity} policy?"` style subqueries

---

## Retrieval Strategy

### Sequential only (v1)

```
for subquery in decomposition.subqueries:
    analysis = analyzer.analyze(subquery.query)
    decision = router.decide(analysis)
    result = hybrid_retriever.retrieve(subquery.query, strategy=decision.strategy)
    subquery_states.append(SubqueryState(...))
```

No `asyncio.gather` until correctness is proven and traced.

### Merge

- RRF across subquery result lists (reuse `ReciprocalRankFusion`)
- Deduplicate by `chunk.id`
- Preserve subquery provenance in trace metadata

---

## Trace / Observability

Named spans for Langfuse alignment:

| Step | Input | Output |
|---|---|---|
| `original_query_analysis` | raw query | `OriginalQueryAnalysis` |
| `query_rewrite` | raw query | `RewriteResult` |
| `resolved_query_analysis` | resolved query | `ResolvedQueryAnalysis` |
| `query_decomposition` | resolved query | `DecompositionResult` |
| `subquery_retrieval_{id}` | `SubQuery` | `HybridRetrievalResult` |
| `subquery_merge` | all subquery hits | merged `ScoredChunk[]` |

---

## Phase 5 Dependency

Phase 5 (rerank → generate → cite) consumes **`DecompositionResult`**, not the original query:

```
SubQuery
    ↓ Retrieve
    ↓ Rerank (per subquery)
    ↓ Generate Partial Answer (per subquery)
    ↓ Merge answers + citations
```

This is cleaner than retrieving everything first and trying to separate attribution afterward.

---

## LangGraph Migration (after 4B)

Move from imperative `RetrievalEngine` orchestration to a branching graph:

```
           Analyze (original)
              │
         Rewrite?
      ┌─────┴─────┐
     Skip      Rewrite
      │           │
      └─────┬─────┘
            ▼
      Analyze (resolved)
            ▼
      Decompose?
      ┌─────┴─────┐
     Skip     Decompose
      │           │
      └─────┬─────┘
            ▼
      Adaptive Router
            ▼
      Hybrid Retrieval
            ▼
      (optional merge node)
```

LangGraph adds value here because rewrite and decompose are conditional branches with distinct downstream paths — not because the engine is insufficient today.

---

## Evaluation Dataset (post-4B, pre-Phase 5)

Build 20–50 representative HR policy questions before adding answer generation.

| Metric | What it measures |
|---|---|
| Router accuracy | Expected vs actual strategy |
| Rewrite accuracy | Was rewrite appropriate? Correct resolved form? |
| Decomposition accuracy | Split only when it should; correct subqueries? |
| Retrieval quality | Recall@k, MRR, dense/sparse overlap |

This gives objective evidence that each pipeline stage improves the system and makes Phase 5+ (reranking, generation, Langfuse, LLM-as-Judge) easier to validate.

Suggested layout:

```
eval/
  dataset.jsonl          # question, expected_strategy, expect_rewrite, expect_decompose, gold_chunk_ids
  run_eval.py            # batch runner against RetrievalEngine
  report.md              # generated metrics summary
```

---

## Implementation Checklist (when approved)

- [ ] Add `SubQuery`, `DecompositionResult`, `QueryDecomposerPort`
- [ ] Implement `RuleBasedQueryDecomposer`
- [ ] Extend `RetrievalEngine` with decomposition branch (sequential)
- [ ] Evolve `SubqueryState` to hold `SubQuery` reference
- [ ] Add merge step with RRF + dedupe
- [ ] Tests: comparison decompose, same-topic no-decompose, end-to-end trace
- [ ] Update `docs/architecture.md` sequence diagram
- [ ] Build eval dataset skeleton

---

## Open Questions — Resolved

| Question | Decision |
|---|---|
| Pass-through | Explicit single-item `SubQuery(id="0", source=pass_through)` |
| Routing | Per-subquery routing via `SubQueryRetrievalPlan` |
| Top-k | Weighted allocation via `TopKAllocator` (LOOKUP=3, FACTUAL=5, SEMANTIC=7) |
| Parallelism | Sequential only in v1 — no `asyncio.gather` |
