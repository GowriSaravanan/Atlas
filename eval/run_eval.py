#!/usr/bin/env python3
"""Run stage-specific evaluation benchmarks for the Adaptive Retrieval Platform."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.fixtures.catalog import build_chunk_catalog
from eval.fixtures.corpus import EVAL_COLLECTION_ID, build_eval_corpus_pdf
from eval.metrics.answer import evaluate_answer_case, summarize_answer
from eval.metrics.confidence import evaluate_confidence_case, summarize_confidence
from eval.metrics.decomposition import evaluate_decomposition_case, summarize_decomposition
from eval.metrics.failure import evaluate_failure_case, summarize_failure
from eval.metrics.latency import summarize_latency
from eval.metrics.report import build_summary_report
from eval.metrics.rerank import evaluate_rerank_case, summarize_rerank
from eval.metrics.retrieval import evaluate_retrieval_case, summarize_retrieval
from eval.metrics.rewrite import evaluate_rewrite_case, summarize_rewrite
from eval.metrics.routing import evaluate_routing_case, summarize_routing

from adaptive_rag.api.dependencies.container import get_container, reset_container
from adaptive_rag.config.settings import get_settings
from adaptive_rag.domain.models.conversation import Message, MessageRole
from adaptive_rag.domain.policies.adaptive_router import AdaptiveRouter
from adaptive_rag.domain.policies.query_analyzer import QueryAnalyzer
from adaptive_rag.domain.policies.query_decomposer import RuleBasedQueryDecomposer
from adaptive_rag.domain.policies.query_rewriter import RuleBasedQueryRewriter
from adaptive_rag.domain.models.retrieval import RetrievalStrategy


DATASETS_DIR = Path(__file__).parent / "datasets"
REPORTS_DIR = Path(__file__).parent / "reports"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL dataset file."""
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def configure_eval_environment(output_dir: Path) -> None:
    """Configure isolated storage and fake embedder for reproducible eval."""
    os.environ["ADAPTIVE_RAG_FAKE_EMBEDDER"] = "1"
    os.environ["ADAPTIVE_RAG_FAKE_RERANKER"] = "1"
    os.environ["ADAPTIVE_RAG_FAKE_LLM"] = "1"
    os.environ["STORAGE__DATA_DIR"] = str(output_dir / "data")
    os.environ["STORAGE__INDEX_DIR"] = str(output_dir / "data" / "indices")
    os.environ["STORAGE__UPLOAD_DIR"] = str(output_dir / "data" / "uploads")
    reset_container()
    get_settings.cache_clear()


def ensure_eval_corpus(output_dir: Path) -> str:
    """Ingest the evaluation corpus and return the collection id."""
    import shutil

    data_dir = output_dir / "data"
    if data_dir.exists():
        shutil.rmtree(data_dir)

    configure_eval_environment(output_dir)
    container = get_container()
    container.ensure_storage_dirs()
    pdf_path = output_dir / "eval_corpus.pdf"
    build_eval_corpus_pdf(pdf_path)
    container.ingest_document_use_case.execute(
        source_path=str(pdf_path),
        collection_id=EVAL_COLLECTION_ID,
    )
    return EVAL_COLLECTION_ID


def parse_context_messages(raw: list[dict[str, str]] | None) -> list[Message]:
    """Convert dataset context payloads to domain messages."""
    if not raw:
        return []
    messages: list[Message] = []
    for item in raw:
        role = MessageRole(item["role"])
        messages.append(Message(role=role, content=item["content"]))
    return messages


def trace_payload(result: Any) -> dict[str, Any]:
    """Serialize trace latency and steps for latency metrics."""
    trace = result.trace
    return {
        "steps": [
            {"step": step.step, "duration_ms": step.duration_ms, "metadata": step.metadata}
            for step in trace.steps
        ],
        "latency_ms": dict(trace.latency_ms),
    }


class EvaluationRunner:
    """Run all benchmark suites against the retrieval platform."""

    def __init__(self, *, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.collection_id = ensure_eval_corpus(output_dir)
        self.container = get_container()
        self.analyzer = QueryAnalyzer()
        self.rewriter = RuleBasedQueryRewriter()
        self.decomposer = RuleBasedQueryDecomposer(self.analyzer)
        self.router = AdaptiveRouter()
        self.catalog = build_chunk_catalog(self.container.index_registry, self.collection_id)
        self.latency_traces: list[dict[str, Any]] = []

    def run_rewrite_eval(self) -> dict[str, Any]:
        cases = []
        for row in load_jsonl(DATASETS_DIR / "rewrite_dataset.jsonl"):
            analysis = self.analyzer.analyze(row["query"])
            rewrite = self.rewriter.rewrite(row["query"], analysis)
            cases.append(
                evaluate_rewrite_case(
                    case_id=row["id"],
                    actual_rewrite=rewrite.rewritten_query,
                    expected_rewrite=row["expected_rewrite"],
                    rewrite_type=row.get("rewrite_type"),
                )
            )
        return {"cases": cases, "summary": summarize_rewrite(cases)}

    def run_routing_eval(self) -> dict[str, Any]:
        cases = []
        for row in load_jsonl(DATASETS_DIR / "routing_dataset.jsonl"):
            analysis = self.analyzer.analyze(row["query"])
            decision = self.router.decide(analysis)
            cases.append(
                evaluate_routing_case(
                    case_id=row["id"],
                    expected_strategy=row["expected_strategy"],
                    actual_strategy=decision.strategy.value,
                    reason=decision.reason,
                )
            )
        return {"cases": cases, "summary": summarize_routing(cases)}

    def run_decomposition_eval(self) -> dict[str, Any]:
        cases = []
        for row in load_jsonl(DATASETS_DIR / "decomposition_dataset.jsonl"):
            analysis = self.analyzer.analyze(row["query"])
            decomposition = self.decomposer.decompose(row["query"], analysis)
            cases.append(
                evaluate_decomposition_case(
                    case_id=row["id"],
                    should_decompose=row["should_decompose"],
                    was_decomposed=decomposition.was_decomposed,
                    actual_subqueries=[subquery.query for subquery in decomposition.subqueries],
                    expected_subqueries=row.get("expected_subqueries"),
                )
            )
        return {"cases": cases, "summary": summarize_decomposition(cases)}

    def run_retrieval_eval(self) -> dict[str, Any]:
        cases = []
        for row in load_jsonl(DATASETS_DIR / "retrieval_dataset.jsonl"):
            top_k = row.get("top_k", 5)
            result = self.container.hybrid_retrieval_use_case.execute(
                query=row["query"],
                collection_id=self.collection_id,
                top_k=top_k,
            )
            self.latency_traces.append(trace_payload(result))
            retrieved_ids = [hit.chunk.id for hit in result.results]
            cases.append(
                evaluate_retrieval_case(
                    case_id=row["id"],
                    retrieved_ids=retrieved_ids,
                    catalog=self.catalog,
                    gold_specs=row.get("gold", []),
                    expected_strategy=row.get("expected_strategy"),
                    actual_strategy=result.strategy.value,
                    top_k=top_k,
                )
            )
        return {"cases": cases, "summary": summarize_retrieval(cases)}

    def run_rerank_eval(self) -> dict[str, Any]:
        cases = []
        for row in load_jsonl(DATASETS_DIR / "retrieval_dataset.jsonl"):
            top_k = row.get("top_k", 5)
            result = self.container.hybrid_retrieval_use_case.execute(
                query=row["query"],
                collection_id=self.collection_id,
                top_k=top_k,
            )
            self.latency_traces.append(trace_payload(result))
            rerank_step = next(
                (step for step in result.trace.steps if step.step == "rerank"),
                None,
            )
            metadata = rerank_step.metadata if rerank_step else {}
            pre_rerank_ids = list(metadata.get("pre_rerank_ids", []))
            post_rerank_ids = [hit.chunk.id for hit in result.results]
            rerank_ms = rerank_step.duration_ms if rerank_step else 0.0
            cases.append(
                evaluate_rerank_case(
                    case_id=row["id"],
                    pre_rerank_ids=pre_rerank_ids,
                    post_rerank_ids=post_rerank_ids,
                    catalog=self.catalog,
                    gold_specs=row.get("gold", []),
                    top_k=top_k,
                    rerank_ms=rerank_ms,
                )
            )
        return {"cases": cases, "summary": summarize_rerank(cases)}

    def run_answer_generation_eval(self) -> dict[str, Any]:
        cases = []
        for row in load_jsonl(DATASETS_DIR / "answer_generation_dataset.jsonl"):
            top_k = row.get("top_k", 5)
            result = self.container.hybrid_retrieval_use_case.execute(
                query=row["query"],
                collection_id=self.collection_id,
                top_k=top_k,
            )
            self.latency_traces.append(trace_payload(result))
            generated = result.generated_answer
            cases.append(
                evaluate_answer_case(
                    case_id=row["id"],
                    answer=generated.answer if generated else "",
                    used_chunk_ids=generated.used_chunk_ids if generated else [],
                    expected_terms=row.get("expected_terms", []),
                    min_chunks_used=row.get("min_chunks_used", 1),
                    latency_ms=generated.latency_ms if generated else 0.0,
                    prompt_tokens=generated.prompt_tokens if generated else 0,
                    completion_tokens=generated.completion_tokens if generated else 0,
                )
            )
        return {"cases": cases, "summary": summarize_answer(cases)}

    def run_confidence_eval(self) -> dict[str, Any]:
        cases = []
        for row in load_jsonl(DATASETS_DIR / "confidence_dataset.jsonl"):
            result = self.container.hybrid_retrieval_use_case.execute(
                query=row["query"],
                collection_id=self.collection_id,
                top_k=5,
            )
            self.latency_traces.append(trace_payload(result))
            gold_ids = set()
            for chunk_id, content, metadata in self.catalog:
                from eval.metrics.retrieval import chunk_matches_gold

                if any(chunk_matches_gold(content, metadata, gold) for gold in row.get("gold", [])):
                    gold_ids.add(chunk_id)
            retrieved_ids = {hit.chunk.id for hit in result.results}
            retrieval_success = bool(retrieved_ids & gold_ids) if gold_ids else len(result.results) > 0
            if row["expected_confidence"] == "low":
                retrieval_success = False
            cases.append(
                evaluate_confidence_case(
                    case_id=row["id"],
                    expected_confidence=row["expected_confidence"],
                    actual_value=result.confidence.value if result.confidence else 0.0,
                    retrieval_success=retrieval_success,
                )
            )
        return {"cases": cases, "summary": summarize_confidence(cases)}

    def run_failure_eval(self) -> dict[str, Any]:
        cases = []
        for row in load_jsonl(DATASETS_DIR / "failure_dataset.jsonl"):
            context = parse_context_messages(row.get("context_messages"))
            result = self.container.hybrid_retrieval_use_case.execute(
                query=row["query"],
                collection_id=self.collection_id,
                context_messages=context or None,
                top_k=5,
            )
            self.latency_traces.append(trace_payload(result))
            cases.append(
                evaluate_failure_case(
                    case_id=row["id"],
                    expected_behavior=row["expected_behavior"],
                    result_count=len(result.results),
                    confidence_value=result.confidence.value if result.confidence else 0.0,
                    confidence_acceptable=result.confidence.is_acceptable if result.confidence else False,
                    was_rewritten=bool(result.rewrite_result and result.rewrite_result.was_rewritten),
                    query_type=result.resolved_analysis.query_type.value if result.resolved_analysis else "unknown",
                )
            )
        return {"cases": cases, "summary": summarize_failure(cases)}

    def run_golden_demo(self) -> dict[str, Any]:
        demos = []
        for row in load_jsonl(DATASETS_DIR / "golden_demo.jsonl"):
            context = parse_context_messages(row.get("context_messages"))
            result = self.container.hybrid_retrieval_use_case.execute(
                query=row["query"],
                collection_id=self.collection_id,
                context_messages=context or None,
                top_k=5,
            )
            self.latency_traces.append(trace_payload(result))
            demos.append(
                {
                    "id": row["id"],
                    "category": row["category"],
                    "description": row["description"],
                    "query": row["query"],
                    "resolved_query": result.resolved_query,
                    "strategy": result.strategy.value,
                    "decomposed": result.decomposition_result.was_decomposed
                    if result.decomposition_result
                    else False,
                    "subqueries": [
                        subquery.query for subquery in result.decomposition_result.subqueries
                    ]
                    if result.decomposition_result
                    else [],
                    "confidence": result.confidence.value if result.confidence else 0.0,
                    "result_count": len(result.results),
                    "trace_steps": [step.step for step in result.trace.steps],
                }
            )
        return {"demos": demos, "count": len(demos)}

    def run_latency_eval(self) -> dict[str, Any]:
        if not self.latency_traces:
            for row in load_jsonl(DATASETS_DIR / "retrieval_dataset.jsonl"):
                result = self.container.hybrid_retrieval_use_case.execute(
                    query=row["query"],
                    collection_id=self.collection_id,
                    top_k=row.get("top_k", 5),
                )
                self.latency_traces.append(trace_payload(result))
        return summarize_latency(self.latency_traces)

    def run_all(self) -> dict[str, Any]:
        sections = {
            "rewrite": self.run_rewrite_eval(),
            "routing": self.run_routing_eval(),
            "decomposition": self.run_decomposition_eval(),
            "retrieval": self.run_retrieval_eval(),
            "rerank": self.run_rerank_eval(),
            "answer_generation": self.run_answer_generation_eval(),
            "confidence": self.run_confidence_eval(),
            "failure": self.run_failure_eval(),
            "golden_demo": self.run_golden_demo(),
        }
        sections["latency"] = self.run_latency_eval()
        return build_summary_report(sections)


def write_report(report: dict[str, Any], output_path: Path) -> None:
    """Write JSON report and a human-readable markdown summary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    markdown_path = output_path.with_suffix(".md")
    lines = [
        "# Evaluation Report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Phase 5 Readiness",
        "",
    ]
    for key, value in report["phase_5_readiness"].items():
        mark = "yes" if value else "no"
        lines.append(f"- {key}: **{mark}**")
    lines.append("")
    lines.append(f"**Instrumentation complete:** {'yes' if report.get('instrumentation_complete') else 'no'}")
    lines.append(f"**Ready for Phase 5:** {'yes' if report['ready_for_phase_5'] else 'no'}")
    lines.append("")
    if report.get("quality_gates"):
        lines.append("## Quality Gates")
        lines.append("")
        for key, value in report["quality_gates"].items():
            lines.append(f"- {key}: **{'pass' if value else 'fail'}**")
        lines.append("")

    sections = report["sections"]
    if "retrieval" in sections:
        summary = sections["retrieval"]["summary"]
        lines.extend(
            [
                "## Retrieval",
                "",
                f"- Recall@k: {summary.get('recall_at_k')}",
                f"- MRR: {summary.get('mrr')}",
                f"- nDCG@k: {summary.get('ndcg_at_k')}",
                "",
            ]
        )
    if "rerank" in sections:
        summary = sections["rerank"]["summary"]
        lines.extend(
            [
                "## Rerank",
                "",
                f"- Pre-rerank Recall@k: {summary.get('pre_recall_at_k')}",
                f"- Post-rerank Recall@k: {summary.get('post_recall_at_k')}",
                f"- Recall delta: {summary.get('recall_delta')}",
                f"- Pre-rerank MRR: {summary.get('pre_mrr')}",
                f"- Post-rerank MRR: {summary.get('post_mrr')}",
                f"- MRR delta: {summary.get('mrr_delta')}",
                f"- Avg rerank latency: {summary.get('avg_rerank_ms')}ms",
                "",
            ]
        )
    if "answer_generation" in sections:
        summary = sections["answer_generation"]["summary"]
        lines.extend(
            [
                "## Answer Generation",
                "",
                f"- Generation success rate: {summary.get('generation_success_rate')}",
                f"- Groundedness rate: {summary.get('groundedness_rate')}",
                f"- Avg latency: {summary.get('avg_latency_ms')}ms",
                f"- Avg prompt tokens: {summary.get('avg_prompt_tokens')}",
                f"- Avg completion tokens: {summary.get('avg_completion_tokens')}",
                "",
            ]
        )
    if "rewrite" in sections:
        summary = sections["rewrite"]["summary"]
        lines.extend(
            [
                "## Rewrite",
                "",
                f"- Exact match rate: {summary.get('exact_match_rate')}",
                "",
            ]
        )
    if "routing" in sections:
        summary = sections["routing"]["summary"]
        lines.extend(
            [
                "## Routing",
                "",
                f"- Router accuracy: {summary.get('router_accuracy')}",
                "",
            ]
        )
    if "decomposition" in sections:
        summary = sections["decomposition"]["summary"]
        lines.extend(
            [
                "## Decomposition",
                "",
                f"- Precision: {summary.get('precision')}",
                f"- Recall: {summary.get('recall')}",
                f"- False decomposition rate: {summary.get('false_decomposition_rate')}",
                "",
            ]
        )
    if "latency" in sections:
        lines.extend(["## Latency", ""])
        for stage, stats in sections["latency"].items():
            lines.append(
                f"- {stage}: avg={stats['avg_ms']}ms p95={stats['p95_ms']}ms p99={stats['p99_ms']}ms"
            )
        lines.append("")

    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Adaptive Retrieval Platform evaluation benchmarks.")
    parser.add_argument(
        "--suite",
        choices=["all", "rewrite", "routing", "decomposition", "retrieval", "rerank", "answer_generation", "confidence", "failure", "golden", "latency"],
        default="all",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORTS_DIR / "run",
        help="Directory for eval artifacts and isolated index storage",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORTS_DIR / "latest.json",
        help="Path for JSON report output",
    )
    args = parser.parse_args()

    runner = EvaluationRunner(output_dir=args.output_dir)

    if args.suite == "all":
        report = runner.run_all()
    else:
        method_name = {
            "rewrite": "run_rewrite_eval",
            "routing": "run_routing_eval",
            "decomposition": "run_decomposition_eval",
            "retrieval": "run_retrieval_eval",
            "rerank": "run_rerank_eval",
            "answer_generation": "run_answer_generation_eval",
            "confidence": "run_confidence_eval",
            "failure": "run_failure_eval",
            "golden": "run_golden_demo",
            "latency": "run_latency_eval",
        }[args.suite]
        section = getattr(runner, method_name)()
        report = build_summary_report({args.suite: section})

    write_report(report, args.report)
    print(json.dumps(report.get("phase_5_readiness", report), indent=2))
    print(f"\nReport written to {args.report}")
    print(f"Markdown summary written to {args.report.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
