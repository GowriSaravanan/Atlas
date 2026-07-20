#!/usr/bin/env python3
"""Run an end-to-end production pipeline benchmark with real AI models."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fitz  # noqa: E402

from adaptive_rag.api.dependencies.container import get_container, reset_container  # noqa: E402
from adaptive_rag.config.settings import get_settings  # noqa: E402


def _use_fake(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes"}


def _build_sample_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "HR Leave Policy", fontsize=18)
    page.insert_text(
        (72, 110),
        "The HR department provides 20 days of annual leave for full-time employees.",
        fontsize=11,
    )
    page.insert_text((72, 150), "Sick Leave", fontsize=16)
    page.insert_text(
        (72, 180),
        "Employees may take up to 10 sick leave days per year with manager approval.",
        fontsize=11,
    )
    document.save(path)
    document.close()


def _component_status() -> dict[str, object]:
    settings = get_settings()
    return {
        "embedder": {
            "implementation": "FakeEmbedder"
            if _use_fake("ADAPTIVE_RAG_FAKE_EMBEDDER")
            else "SentenceTransformerEmbedder",
            "model": settings.embedding.model_name,
            "fake": _use_fake("ADAPTIVE_RAG_FAKE_EMBEDDER"),
        },
        "reranker": {
            "implementation": "FakeReranker"
            if _use_fake("ADAPTIVE_RAG_FAKE_RERANKER")
            else "CrossEncoderReranker",
            "model": settings.reranker.model_name,
            "fake": _use_fake("ADAPTIVE_RAG_FAKE_RERANKER"),
        },
        "llm": {
            "implementation": "FakeLLM"
            if _use_fake("ADAPTIVE_RAG_FAKE_LLM")
            else (
                "OpenRouterProviderLLM"
                if settings.llm.provider == "openrouter"
                else "ProviderLLM"
            ),
            "provider": settings.llm.provider,
            "model": settings.llm.model,
            "fake": _use_fake("ADAPTIVE_RAG_FAKE_LLM"),
            "api_key_present": bool(settings.llm.openrouter_api_key)
            if settings.llm.provider == "openrouter"
            else True,
        },
        "citation_formatter": {
            "implementation": "EvidenceCitationFormatter",
            "fake": False,
        },
    }


def run_benchmark(*, output_dir: Path) -> dict[str, object]:
    """Execute ingestion → retrieval → rerank → answer → citation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("STORAGE__DATA_DIR", str(output_dir / "data"))
    os.environ.setdefault("STORAGE__INDEX_DIR", str(output_dir / "data" / "indices"))
    os.environ.setdefault("STORAGE__UPLOAD_DIR", str(output_dir / "data" / "uploads"))

    reset_container()
    get_settings.cache_clear()

    components = _component_status()
    settings = get_settings()
    container = get_container()
    container.ensure_storage_dirs()

    pdf_path = output_dir / "benchmark_corpus.pdf"
    _build_sample_pdf(pdf_path)

    timings: dict[str, float] = {}

    start = time.perf_counter()
    ingest_result = container.ingest_document_use_case.execute(
        source_path=str(pdf_path),
        collection_id="benchmark",
    )
    timings["ingestion_ms"] = round((time.perf_counter() - start) * 1000, 2)

    embedder = container.embedder
    timings["embedding_dimension"] = embedder.dimension
    timings["embedding_model"] = embedder.model_name

    query = "How many sick leave days are allowed per year?"
    pipeline_error = None
    result = None
    start = time.perf_counter()
    try:
        result = container.hybrid_retrieval_use_case.execute(
            query=query,
            collection_id="benchmark",
            top_k=5,
        )
    except Exception as exc:  # noqa: BLE001 - benchmark should capture runtime failures
        pipeline_error = f"{type(exc).__name__}: {exc}"
    timings["retrieval_pipeline_ms"] = round((time.perf_counter() - start) * 1000, 2)

    if result is None:
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "components": components,
            "settings_snapshot": {
                "embedding_model": settings.embedding.model_name,
                "reranker_model": settings.reranker.model_name,
                "llm_provider": settings.llm.provider,
                "llm_model": settings.llm.model,
            },
            "ingestion": {
                "document_id": ingest_result.document_id,
                "chunk_count": ingest_result.chunk_count,
                "latency_ms": timings["ingestion_ms"],
            },
            "timings_ms": timings,
            "pipeline_error": pipeline_error,
            "example_output": {"query": query},
        }

    timings.update({key: round(value, 2) for key, value in result.trace.latency_ms.items()})

    rerank_step = next((step for step in result.trace.steps if step.step == "rerank"), None)
    answer_step = next(
        (step for step in result.trace.steps if step.step == "answer_generation"),
        None,
    )
    citation_step = next(
        (step for step in result.trace.steps if step.step == "citation_formatting"),
        None,
    )

    generated = result.generated_answer
    example = {
        "query": query,
        "resolved_query": result.resolved_query,
        "strategy": result.strategy.value,
        "confidence": result.confidence.value if result.confidence else None,
        "retrieved_chunks": len(result.results),
        "rerank_skipped": rerank_step.metadata.get("skipped") if rerank_step else None,
        "pre_rerank_ids": rerank_step.metadata.get("pre_rerank_ids") if rerank_step else [],
        "post_rerank_ids": rerank_step.metadata.get("post_rerank_ids") if rerank_step else [],
        "answer": generated.answer if generated else None,
        "used_chunk_ids": generated.used_chunk_ids if generated else [],
        "citations": [
            citation.model_dump(mode="json") for citation in (generated.citations if generated else [])
        ],
        "citation_formats_preview": {
            "markdown": (
                generated.citation_formats.markdown[:500] + "..."
                if generated and generated.citation_formats and len(generated.citation_formats.markdown) > 500
                else (generated.citation_formats.markdown if generated and generated.citation_formats else None)
            ),
        },
    }

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "components": components,
        "settings_snapshot": {
            "embedding_model": settings.embedding.model_name,
            "reranker_model": settings.reranker.model_name,
            "llm_provider": settings.llm.provider,
            "llm_model": settings.llm.model,
        },
        "ingestion": {
            "document_id": ingest_result.document_id,
            "chunk_count": ingest_result.chunk_count,
            "latency_ms": timings["ingestion_ms"],
        },
        "timings_ms": timings,
        "pipeline_steps": {
            "rerank": rerank_step.metadata if rerank_step else None,
            "answer_generation": answer_step.metadata if answer_step else None,
            "citation_formatting": citation_step.metadata if citation_step else None,
        },
        "example_output": example,
    }


def write_report(report: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    components = report["components"]
    lines = [
        "# Production Models Benchmark",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Component Status",
        "",
        "| Component | Implementation | Model | Fake? |",
        "|---|---|---|---|",
    ]
    for name, info in components.items():
        model = info.get("model", "—")
        lines.append(
            f"| {name} | {info['implementation']} | {model} | {'yes' if info['fake'] else 'no'} |"
        )

    lines.extend(["", "## Timings (ms)", ""])
    for key, value in report["timings_ms"].items():
        lines.append(f"- {key}: {value}")

    example = report["example_output"]
    lines.extend(
        [
            "",
            "## Example Output",
            "",
            f"**Query:** {example['query']}",
            "",
            f"**Answer:** {example['answer']}",
            "",
            f"**Citations:** {len(example['citations'])}",
            "",
        ]
    )
    output_path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    output_dir = ROOT / "eval" / "reports" / "production-run"
    report_path = ROOT / "eval" / "reports" / "production-benchmark.json"
    report = run_benchmark(output_dir=output_dir)
    write_report(report, report_path)
    print(json.dumps(report["components"], indent=2))
    print(f"\nReport written to {report_path}")
    print(f"Markdown summary written to {report_path.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
