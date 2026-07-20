# Production Models Milestone — Report

Generated after wiring real AI adapters for production use.

## Summary

| Layer | Production implementation | Default model | Unit tests |
|---|---|---|---|
| **Embedder** | `SentenceTransformerEmbedder` | `BAAI/bge-base-en-v1.5` | `FakeEmbedder` (`ADAPTIVE_RAG_FAKE_EMBEDDER=1`) |
| **Reranker** | `CrossEncoderReranker` | `BAAI/bge-reranker-base` | `FakeReranker` (`ADAPTIVE_RAG_FAKE_RERANKER=1`) |
| **LLM** | `OpenRouterProviderLLM` | `OPENROUTER_MODEL` (default: `meta-llama/llama-3.1-8b-instruct`) | `FakeLLM` (`ADAPTIVE_RAG_FAKE_LLM=1`) |
| **Citation** | `EvidenceCitationFormatter` | N/A (rule-based) | Same (no fake) |

## OpenRouter configuration

Yes — **full end-to-end answer generation requires `OPENROUTER_API_KEY`**.

Set in local `.env` (gitignored):

```bash
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
RERANKER_MODEL=BAAI/bge-reranker-base
LLM__PROVIDER=openrouter
```

Do **not** commit API keys to the repository. Copy from `.env.example` and fill in locally.

## Test matrix

| Suite | Command | Adapters |
|---|---|---|
| Unit tests | `uv run pytest -m "not integration"` | All fakes (via `conftest.py`) |
| Integration | `uv run pytest -m integration` | Real embedder + reranker; OpenRouter when key in `.env` |
| Production benchmark | `uv run python scripts/run_production_benchmark.py` | Real adapters from `.env` |

### Test results (local run)

- **86 unit tests passed** (fake adapters)
- **8 integration tests passed** including live OpenRouter full pipeline
- **Production benchmark completed** — see `eval/reports/production-benchmark.json`

| Stage | Latency (ms) |
|---|---|
| Ingestion (embed + index) | 9,020 |
| Full retrieval pipeline | 15,467 |
| Dense retrieval | 354 |
| Rerank (BGE reranker) | 5,660 |
| Answer generation (OpenRouter) | 4,320 |
| Citation formatting | 0.5 |

**Example query:** `How many sick leave days are allowed per year?`

**Answer (OpenRouter):** `10 sick leave days per year` (grounded in Sick Leave chunk, confidence 0.999)

**Citations:** 2 chunks attached with structured metadata and formats.

## Pipeline stages (production)

```
Ingest PDF
  → chunk + metadata
  → SentenceTransformerEmbedder (BGE)
  → FAISS dense + BM25 sparse index

Query
  → hybrid retrieval
  → CrossEncoderReranker (BGE reranker)
  → ContextBuilder + OpenRouterProviderLLM
  → EvidenceCitationFormatter
```

## Example output shape

After a successful `/retrieve` or benchmark run:

```json
{
  "query": "How many sick leave days are allowed?",
  "generated_answer": {
    "answer": "...",
    "used_chunk_ids": ["..."],
    "citations": [
      {
        "chunk_id": "...",
        "document_id": "...",
        "page_number": 1,
        "section_title": "Sick Leave",
        "confidence": 0.92,
        "excerpt": "Employees may take up to 10 sick leave days..."
      }
    ],
    "citation_formats": {
      "markdown": "...",
      "plain_text": "...",
      "json_text": "..."
    }
  },
  "trace": {
    "steps": ["answer_generation", "citation_formatting", "rerank", "..."]
  }
}
```

## Benchmark artifacts

After running the benchmark script:

- `eval/reports/production-benchmark.json`
- `eval/reports/production-benchmark.md`

## Security note

If an API key was shared in chat or committed by mistake, **rotate it immediately** in the OpenRouter dashboard and update local `.env` only.
