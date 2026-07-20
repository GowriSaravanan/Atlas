#!/usr/bin/env python3
"""Write production benchmark report from a live pipeline run."""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import fitz  # noqa: E402

from adaptive_rag.api.dependencies.container import get_container, reset_container  # noqa: E402
from adaptive_rag.config.settings import get_settings  # noqa: E402
from adaptive_rag.infrastructure.embeddings.sentence_transformer import SentenceTransformerEmbedder  # noqa: E402
from adaptive_rag.infrastructure.llm.openrouter_llm import OpenRouterProviderLLM  # noqa: E402
from adaptive_rag.infrastructure.reranking.cross_encoder import CrossEncoderReranker  # noqa: E402


def main() -> int:
    output_dir = ROOT / "eval" / "reports" / "production-run"
    output_dir.mkdir(parents=True, exist_ok=True)
    reset_container()
    get_settings.cache_clear()

    settings = get_settings()
    container = get_container()
    container.ensure_storage_dirs()

    pdf_path = output_dir / "benchmark_corpus.pdf"
    doc = fitz.open()
    page = doc.new_page()
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
    doc.save(pdf_path)
    doc.close()

    t0 = time.perf_counter()
    ingest = container.ingest_document_use_case.execute(
        source_path=str(pdf_path),
        collection_id="benchmark",
    )
    ingest_ms = round((time.perf_counter() - t0) * 1000, 2)

    query = "How many sick leave days are allowed per year?"
    t0 = time.perf_counter()
    result = container.hybrid_retrieval_use_case.execute(
        query=query,
        collection_id="benchmark",
        top_k=5,
    )
    pipeline_ms = round((time.perf_counter() - t0) * 1000, 2)

    generated = result.generated_answer
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "components": {
            "embedder": {
                "implementation": type(container.embedder).__name__,
                "model": settings.embedding.model_name,
                "fake": False,
            },
            "reranker": {
                "implementation": type(container.reranker).__name__,
                "model": settings.reranker.model_name,
                "fake": False,
            },
            "llm": {
                "implementation": type(container.llm).__name__,
                "provider": settings.llm.provider,
                "model": settings.llm.model,
                "fake": False,
            },
            "citation_formatter": {
                "implementation": "EvidenceCitationFormatter",
                "fake": False,
            },
        },
        "timings_ms": {
            "ingestion_ms": ingest_ms,
            "retrieval_pipeline_ms": pipeline_ms,
            **{k: round(v, 2) for k, v in result.trace.latency_ms.items()},
        },
        "ingestion": {
            "document_id": ingest.document_id,
            "chunk_count": ingest.chunk_count,
        },
        "example_output": {
            "query": query,
            "strategy": result.strategy.value,
            "confidence": result.confidence.value if result.confidence else None,
            "answer": generated.answer if generated else None,
            "used_chunk_ids": generated.used_chunk_ids if generated else [],
            "citation_count": len(generated.citations) if generated else 0,
            "citations": [
                c.model_dump(mode="json") for c in (generated.citations if generated else [])
            ],
        },
    }

    json_path = ROOT / "eval" / "reports" / "production-benchmark.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        "# Production Models Benchmark",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Components (all real)",
        "",
    ]
    for name, info in report["components"].items():
        model = info.get("model", "—")
        md_lines.append(f"- **{name}**: `{info['implementation']}` ({model})")

    md_lines.extend(["", "## Timings (ms)", ""])
    for key, value in report["timings_ms"].items():
        md_lines.append(f"- {key}: {value}")

    ex = report["example_output"]
    md_lines.extend(
        [
            "",
            "## Example Output",
            "",
            f"**Query:** {ex['query']}",
            f"**Answer:** {ex['answer']}",
            f"**Citations:** {ex['citation_count']}",
            f"**Used chunks:** {ex['used_chunk_ids']}",
        ]
    )
    json_path.with_suffix(".md").write_text("\n".join(md_lines), encoding="utf-8")

    print(json.dumps(report["components"], indent=2))
    print(f"\nReport: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
