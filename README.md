# Adaptive Hybrid RAG Platform

Production-quality portfolio project demonstrating enterprise AI engineering with clean architecture, LangGraph orchestration, and hybrid retrieval.

## Phase 0 — Project Skeleton

Phase 0 provides the project structure, domain models, port interfaces, DI container, empty LangGraph workflows, and smoke tests. No RAG logic yet.

## Quick Start

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Start API server
uv run adaptive-rag
```

## API

- `GET /health` — health check
- `POST /api/v1/query` — RAG query (skeleton)
- `POST /api/v1/ingest` — document ingestion (skeleton)

## Configuration

Copy `.env.example` to `.env` and adjust settings. See `src/adaptive_rag/config/settings.py` for all options.
