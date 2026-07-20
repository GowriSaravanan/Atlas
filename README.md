# Adaptive Hybrid RAG Platform

Production-quality portfolio project demonstrating enterprise AI engineering with clean architecture, LangGraph orchestration, and hybrid retrieval.

## Phase 1 — Ingestion Pipeline

Phase 1 implements intelligent PDF ingestion, adaptive chunking, metadata extraction, and hybrid indexing (FAISS + BM25).

### Features

- **PyMuPDF loader** with heading detection and section extraction
- **Adaptive chunker** (structure-aware with token fallback)
- **Rule-based metadata extraction** (department, dates, titles)
- **FAISS dense index** + **BM25 sparse index** with disk persistence
- **LangGraph ingest workflow**: load → extract metadata → chunk → index
- **Upload API** for PDF ingestion

## Phase 3 — Adaptive Retrieval

Phase 3 adds query analysis, adaptive routing, automatic metadata scoping, and confidence scoring via `RetrievalEngine`.

### Pipeline

```
Query → QueryAnalyzer → AdaptiveRouter → MetadataScopeBuilder
     → HybridRetriever → ConfidenceScorer → Result + Trace
```

Omit `strategy` in the request to enable adaptive routing.

### Features

- **Rule-based query analysis** (policy IDs, departments, years, intent)
- **Adaptive router** selects BM25 / dense / hybrid automatically
- **Metadata scope builder** derives filters from analysis
- **Confidence scoring** with explainable breakdown

## Phase 4A — Query Rewriting

Conditional pre-retrieval rewriting when `rewrite_decision.should_rewrite` is true.

### Pipeline

```
Query → Original Analysis → [Rewrite?] → Resolved Analysis → Decompose → Retrieve
```

### Features

- **Rule-based query rewriter** for follow-up phrasing (`"What about maternity leave?"` → `"What is the maternity leave policy?"`)
- **Context-aware rewrite** via optional `context_messages` on `RetrievalRequest`
- **`original_analysis` / `resolved_analysis`** with named trace steps for observability

## Phase 4B — Query Decomposition

Sequential subquery retrieval with per-subquery routing and weighted top-k allocation.

### Pipeline

```
Resolved Analysis → Decompose → for each SubQuery: analyze → route → retrieve → merge (RRF)
```

### Features

- **`SubQuery`** with `id`, `entity`, `source`, `query_type` — always at least one (pass-through)
- **`DecompositionResult`** and **`SubQueryResult`** for Phase 5 per-subquery rerank/generate
- **`SubQueryRetrievalPlan`** — per-subquery strategy, reason, and top-k budget
- **`TopKAllocator`** — weighted budget (LOOKUP=3, FACTUAL=5, SEMANTIC=7)
- **Sequential retrieval only** — no parallel execution in v1

## Evaluation Framework

Retrieval features are **frozen** until benchmarks pass. See [docs/evaluation.md](docs/evaluation.md).

```bash
uv run python eval/run_eval.py
uv run python eval/run_eval.py --suite golden
```

Stage-specific datasets cover rewrite, routing, decomposition, retrieval, confidence, failure cases, and a 10-query golden demo for interviews.

## Quick Start

```bash
uv sync
uv run pytest
uv run adaptive-rag
```

## API

- `GET /health`
- `POST /api/v1/query` — RAG query (skeleton, Phase 4+)
- `POST /api/v1/ingest` — ingest from local file path
- `POST /api/v1/ingest/upload` — upload PDF for ingestion
- `GET /api/v1/collections/{collection_id}/stats` — index statistics
- `POST /api/v1/retrieve` — hybrid retrieval (BM25 / dense / hybrid)

### Retrieve example

```bash
curl -X POST "http://localhost:8000/api/v1/retrieve" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the annual leave policy?",
    "collection_id": "default",
    "strategy": "hybrid",
    "top_k": 5
  }'
```

### Upload example

```bash
curl -X POST "http://localhost:8000/api/v1/ingest/upload" \
  -F "file=@document.pdf" \
  -F "collection_id=default"
```

## Configuration

Copy `.env.example` to `.env`. Key settings:

- `CHUNKING__MAX_TOKENS`, `CHUNKING__STRATEGY`
- `STORAGE__INDEX_DIR`, `STORAGE__UPLOAD_DIR`
- `EMBEDDING__MODEL_NAME`
- `ADAPTIVE_RAG_FAKE_EMBEDDER=1` — use deterministic fake embedder (tests)

## Architecture Notes

- Domain policies (`AdaptiveChunker`, `DocumentMetadataExtractor`) contain business logic
- Infrastructure adapters implement ports (PyMuPDF, FAISS, BM25, Sentence Transformers)
- `CollectionIndexRegistry` manages per-collection indexes with persistence

## Platform Compatibility

On older macOS x86_64 hosts, the project pins compatible versions of `faiss-cpu` and `torch`. Use `uv sync` to install resolved dependencies.
