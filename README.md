<div align="center">

# Atlas

### Adaptive Hybrid Retrieval-Augmented Generation Platform

*A production-inspired AI engineering project demonstrating adaptive retrieval, Clean Architecture, and evaluation-driven development.*

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)
![Architecture](https://img.shields.io/badge/Architecture-Clean%20Architecture-orange.svg)
![Tests](https://img.shields.io/badge/Tests-94%20Passing-success.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

</div>

---

# Overview

Atlas is an adaptive Retrieval-Augmented Generation (RAG) platform built to demonstrate modern AI engineering practices rather than framework-centric development.

The project focuses on designing a modular retrieval pipeline capable of handling complex search scenarios through adaptive routing, hybrid retrieval, query rewriting, query decomposition, reranking, and evidence-grounded answer generation.

Instead of coupling business logic to AI frameworks, Atlas follows **Clean Architecture** and **Ports & Adapters**, allowing infrastructure components—including language models, vector stores, retrieval algorithms, and APIs—to evolve independently.

The result is a maintainable, testable, and production-inspired retrieval system that emphasizes software engineering principles as much as retrieval quality.

---

# Why Atlas?

Most public RAG repositories demonstrate how to integrate an orchestration framework.

Atlas focuses on **how a retrieval system should be engineered.**

The project emphasizes:

- Retrieval engineering
- Modular architecture
- Explainable retrieval decisions
- Framework independence
- Evaluation-driven development
- Production-oriented API design

Rather than building the largest possible RAG pipeline, Atlas demonstrates how modern AI systems can be designed using maintainable software architecture.

---

# Key Features

## Adaptive Retrieval

- Adaptive Retrieval Strategy Selection
- Dense Retrieval
- BM25 Retrieval
- Hybrid Retrieval
- Reciprocal Rank Fusion (RRF)
- Metadata Filtering

---

## Query Understanding

- Query Analysis
- Query Rewriting
- Query Decomposition
- Retrieval Strategy Selection
- Context Optimization

---

## Ranking

- Cross-Encoder Reranking
- Confidence Scoring
- Context Ranking
- Top-K Selection

---

## Answer Generation

- Evidence-Grounded Prompt Construction
- Structured Citation Formatting
- Configurable Prompt Templates
- Response Metadata

---

## Engineering

- Clean Architecture
- Ports & Adapters
- Dependency Injection
- FastAPI
- Docker
- GitHub Actions
- Comprehensive Unit Testing
- Fake Infrastructure Adapters

---

## Evaluation

- Retrieval Evaluation
- Query Rewrite Evaluation
- Routing Evaluation
- Citation Validation
- Golden Dataset Testing
- Production Benchmark Scripts

---

# System Architecture

```text
                          User Query
                               │
                               ▼
                 Query Analysis & Classification
                               │
          ┌────────────────────┴────────────────────┐
          │                                         │
          ▼                                         ▼
  Query Rewriting                         Query Decomposition
          │                                         │
          └────────────────────┬────────────────────┘
                               ▼
                 Adaptive Retrieval Router
                               │
      ┌────────────────────────┼────────────────────────┐
      │                        │                        │
      ▼                        ▼                        ▼
 BM25 Retrieval        Dense Retrieval        Hybrid Retrieval
                                                   │
                                                   ▼
                                 Reciprocal Rank Fusion
                                                   │
                                                   ▼
                                  Cross Encoder Reranker
                                                   │
                                                   ▼
                                     Context Construction
                                                   │
                                                   ▼
                                      Answer Generation
                                                   │
                                                   ▼
                                     Citation Formatting
                                                   │
                                                   ▼
                                          Final Response
```

---

# Clean Architecture

Atlas follows a layered architecture that separates business logic from infrastructure.

```text
                    Presentation Layer
                FastAPI REST Endpoints
                         │
                         ▼
                Application Use Cases
                         │
                         ▼
              Domain Models & Business Rules
                         │
                         ▼
               Ports (Interfaces / Contracts)
                         │
                         ▼
             Infrastructure Implementations
```

Business logic has no direct dependency on:

- FastAPI
- FAISS
- OpenRouter
- LangGraph
- Docker

Infrastructure is accessed through adapters implementing application-defined ports.

---

# Repository Structure

```text
atlas/
│
├── src/
│   ├── api/
│   ├── application/
│   ├── config/
│   ├── domain/
│   ├── infrastructure/
│   └── main.py
│
├── docs/
│
├── eval/
│
├── scripts/
│
├── tests/
│
├── docker/
│
├── .github/
│
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

# Technology Stack

| Category | Technology |
|-----------|------------|
| Language | Python 3.12 |
| API | FastAPI |
| Architecture | Clean Architecture |
| Pattern | Ports & Adapters |
| Dependency Injection | Python DI |
| Sparse Retrieval | BM25 |
| Dense Retrieval | FAISS |
| Embeddings | BAAI/bge-base-en-v1.5 |
| Reranker | BAAI/bge-reranker-base |
| LLM Provider | OpenRouter |
| Workflow | LangGraph (Document Ingestion) |
| PDF Processing | PyMuPDF |
| Testing | Pytest |
| Containerization | Docker |
| CI/CD | GitHub Actions |

---

# Retrieval Pipeline

## 1. Query Analysis

Every incoming query is analysed to determine:

- query complexity
- retrieval intent
- routing strategy
- decomposition requirements

---

## 2. Query Rewriting

Ambiguous user queries are rewritten into retrieval-friendly queries.

Example

Input

```text
What's the waiting period?
```

Rewritten

```text
What is the waiting period for pre-existing diseases under the policy?
```

---

## 3. Query Decomposition

Complex questions are decomposed into smaller retrieval tasks.

Example

```text
Explain maternity leave and insurance eligibility.
```

becomes

```text
Subquery 1

Explain maternity leave.

Subquery 2

Explain insurance eligibility.
```

---

## 4. Adaptive Routing

Atlas dynamically selects an appropriate retrieval strategy.

Supported strategies

- Dense Retrieval
- BM25 Retrieval
- Hybrid Retrieval

---

## 5. Hybrid Retrieval

Atlas combines lexical and semantic search using Reciprocal Rank Fusion (RRF).

Benefits include:

- improved recall
- reduced retrieval bias
- complementary ranking signals

---

## 6. Cross Encoder Reranking

Retrieved passages are reranked using a transformer-based cross encoder to improve semantic relevance before answer generation.

---

## 7. Context Construction

The highest ranked passages are assembled into a structured prompt while preserving metadata including:

- source document
- page number
- chunk identifier

---

## 8. Answer Generation

The language model generates responses using only retrieved evidence.

This helps reduce unsupported or hallucinated responses.

---

## 9. Citation Formatting

Supporting evidence is formatted into structured citations before returning the final response.

Supported formats:

- Markdown
- Plain Text
- JSON

---

# Getting Started

## Prerequisites

- Python 3.12+
- uv
- Docker (optional)
- OpenRouter API Key

---

## Clone Repository

```bash
git clone https://github.com/<username>/atlas.git

cd atlas
```

---

## Install Dependencies

```bash
uv sync
```

---

## Configure Environment

Copy the example configuration.

```bash
cp .env.example .env
```

Update the required environment variables.

---

## Run Tests

```bash
uv run pytest
```

---

## Start Development Server

```bash
uv run adaptive-rag
```

or

```bash
uv run python -m src.main
```

---

## Docker

Build the application.

```bash
docker compose build
```

Run services.

```bash
docker compose up
```

---

# API Overview

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | `/health` | Liveness check |
| GET | `/ready` | Readiness check |
| POST | `/api/v1/retrieve` | Adaptive Hybrid Retrieval |
| POST | `/api/v1/ingest` | Document ingestion |
| POST | `/api/v1/ingest/upload` | Upload PDF |
| GET | `/api/v1/collections/{collection_id}/stats` | Collection statistics |

---

# Example Request

```http
POST /api/v1/retrieve
```

```json
{
  "query": "What is the waiting period for pre-existing diseases?",
  "collection_id": "insurance-policy",
  "top_k": 5
}
```

---

# Example Response

```json
{
  "answer": "...",
  "citations": [
    {
      "document": "policy.pdf",
      "page": 12
    }
  ],
  "latency_ms": 842,
  "retrieval_strategy": "hybrid"
}
```

---
# Configuration

Atlas uses environment variables for configuring language models,
retrieval components, storage paths, and runtime settings.

Create a local configuration file.

```bash
cp .env.example .env
```

Update the required values before starting the application.

---

# Testing

Atlas includes automated tests covering application logic,
retrieval components, infrastructure adapters, and API endpoints.

Run all tests

```bash
uv run pytest
```

Run only unit tests

```bash
uv run pytest -m "not integration"
```

Run integration tests

```bash
uv run pytest -m integration
```

---

# Evaluation

Atlas follows an evaluation-driven development approach.

Each retrieval stage can be validated independently to measure
pipeline quality and identify regressions.

Current evaluation includes:

- Retrieval Quality
- Query Rewriting
- Query Routing
- Query Decomposition
- Cross-Encoder Reranking
- Citation Formatting
- Golden Dataset Evaluation
- Production Benchmark Scripts

Evaluation reports are generated under:

```text
eval/reports/
```

---

# Design Principles

Atlas is built around a small set of engineering principles.

## Separation of Concerns

Business logic remains independent of infrastructure.

External systems communicate through adapters rather than directly
coupling application logic to implementation details.

---

## Retrieval Before Generation

Answer quality depends on retrieval quality.

Rather than relying solely on prompt engineering,
Atlas prioritizes improving retrieval through query understanding,
adaptive routing, reranking, and evidence selection.

---

## Evaluation First

New functionality should be measurable.

Retrieval improvements are evaluated before becoming
part of the default pipeline.

---

## Maintainability

The project is organised to allow retrieval algorithms,
language models, vector stores, and infrastructure
to evolve independently.

---

# Project Status

Current implementation includes:

- Adaptive Hybrid Retrieval
- Query Rewriting
- Query Decomposition
- Reciprocal Rank Fusion
- Cross-Encoder Reranking
- Citation Formatting
- FastAPI API
- Docker Support
- GitHub Actions CI
- Evaluation Framework

---

# Limitations

Atlas intentionally keeps several areas simple.

Current limitations include:

- Sequential execution for decomposed queries
- Rule-based query analysis
- Single-node deployment
- No authentication or authorization layer
- Citation verification is not implemented

These trade-offs keep the project focused on retrieval engineering
while leaving room for future extensions.

---

# Roadmap

Potential future enhancements include:

- Streaming responses
- LLM-assisted query analysis
- Multi-vector retrieval
- Additional language model providers
- Observability dashboards
- Authentication and user management

---

# Contributing

Contributions are welcome.

Please open an issue before submitting significant changes
to discuss the proposed approach.

---

# License

This project is licensed under the MIT License.

---

# Acknowledgements

Atlas was built as a portfolio project to explore
production-inspired AI engineering practices including
retrieval engineering, clean architecture,
evaluation-driven development, and maintainable software design.
