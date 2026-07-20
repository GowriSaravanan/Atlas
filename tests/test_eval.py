"""Evaluation framework tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.metrics.answer import evaluate_answer_case, summarize_answer
from eval.metrics.citation import evaluate_citation_case, summarize_citation
from eval.metrics.decomposition import evaluate_decomposition_case, summarize_decomposition
from eval.metrics.latency import summarize_latency
from eval.metrics.rerank import evaluate_rerank_case, summarize_rerank
from eval.metrics.retrieval import evaluate_retrieval_case, mrr, recall_at_k
from eval.metrics.rewrite import evaluate_rewrite_case, summarize_rewrite
from eval.metrics.routing import evaluate_routing_case, summarize_routing


def test_recall_and_mrr() -> None:
    retrieved = ["a", "b", "c", "d"]
    gold = {"b", "d"}
    assert recall_at_k(retrieved, gold, 3) == 0.5
    assert mrr(retrieved, gold) == 0.5


def test_rewrite_exact_match_summary() -> None:
    cases = [
        evaluate_rewrite_case(
            case_id="RW001",
            actual_rewrite="What is the maternity leave policy?",
            expected_rewrite="What is the maternity leave policy?",
        ),
        evaluate_rewrite_case(
            case_id="RW002",
            actual_rewrite="What is the paternity leave policy?",
            expected_rewrite="What is the maternity leave policy?",
        ),
    ]
    summary = summarize_rewrite(cases)
    assert summary["exact_match_rate"] == 0.5


def test_routing_accuracy_summary() -> None:
    cases = [
        evaluate_routing_case(case_id="RT001", expected_strategy="bm25", actual_strategy="bm25", reason=""),
        evaluate_routing_case(case_id="RT002", expected_strategy="hybrid", actual_strategy="dense", reason=""),
    ]
    summary = summarize_routing(cases)
    assert summary["router_accuracy"] == 0.5


def test_citation_metrics() -> None:
    case = evaluate_citation_case(
        case_id="CT001",
        used_chunk_ids=["c1", "c2"],
        citation_chunk_ids=["c1", "c2"],
        min_citations=1,
    )
    summary = summarize_citation([case])
    assert case["coverage"] == 1.0
    assert case["precision"] == 1.0
    assert case["missing_citation_rate"] == 0.0
    assert case["invalid_citation_rate"] == 0.0
    assert summary["citation_coverage"] == 1.0


def test_decomposition_false_decomposition_rate() -> None:
    cases = [
        evaluate_decomposition_case(
            case_id="D1",
            should_decompose=False,
            was_decomposed=True,
            actual_subqueries=["a", "b"],
        ),
        evaluate_decomposition_case(
            case_id="D2",
            should_decompose=True,
            was_decomposed=True,
            actual_subqueries=["a", "b"],
            expected_subqueries=["a", "b"],
        ),
    ]
    summary = summarize_decomposition(cases)
    assert summary["false_decomposition_rate"] == 0.5
    assert summary["precision"] == 0.5
    assert summary["recall"] == 1.0


def test_latency_summary() -> None:
    traces = [
        {
            "steps": [
                {"step": "original_query_analysis", "duration_ms": 8.0},
                {"step": "query_rewrite", "duration_ms": 2.0},
                {"step": "subquery_retrieval_0", "duration_ms": 48.0},
                {"step": "subquery_merge", "duration_ms": 3.0},
            ],
            "latency_ms": {},
        }
    ]
    summary = summarize_latency(traces)
    assert summary["analysis"]["avg_ms"] == 8.0
    assert summary["rewrite"]["avg_ms"] == 2.0
    assert summary["retrieval"]["avg_ms"] == 48.0


def test_rerank_before_after_metrics() -> None:
    catalog = [
        ("c1", "Policy HR-203 grants annual leave", {"section_title": "Policy HR-203: Annual Leave"}),
        ("c2", "Policy HR-105 sick leave", {"section_title": "Policy HR-105: Sick Leave"}),
    ]
    case = evaluate_rerank_case(
        case_id="RR001",
        pre_rerank_ids=["c2", "c1"],
        post_rerank_ids=["c1", "c2"],
        catalog=catalog,
        gold_specs=[{"policy_id": "HR-203"}],
        top_k=2,
        rerank_ms=12.5,
    )
    assert case["pre_mrr"] == 0.5
    assert case["post_mrr"] == 1.0
    assert case["mrr_delta"] == 0.5
    assert case["rank_changed"] is True


def test_rerank_summary() -> None:
    cases = [
        evaluate_rerank_case(
            case_id="RR001",
            pre_rerank_ids=["c2", "c1"],
            post_rerank_ids=["c1", "c2"],
            catalog=[("c1", "gold", {}), ("c2", "other", {})],
            gold_specs=[{"content_contains": "gold"}],
            top_k=2,
            rerank_ms=10.0,
        ),
        evaluate_rerank_case(
            case_id="RR002",
            pre_rerank_ids=["c1", "c2"],
            post_rerank_ids=["c1", "c2"],
            catalog=[("c1", "gold", {}), ("c2", "other", {})],
            gold_specs=[{"content_contains": "gold"}],
            top_k=2,
            rerank_ms=20.0,
        ),
    ]
    summary = summarize_rerank(cases)
    assert summary["count"] == 2
    assert summary["mrr_delta"] == 0.25
    assert summary["avg_rerank_ms"] == 15.0


def test_answer_generation_metrics() -> None:
    case = evaluate_answer_case(
        case_id="AG001",
        answer="Employees may take up to 10 sick leave days per year.",
        used_chunk_ids=["c1"],
        expected_terms=["10", "sick"],
        min_chunks_used=1,
        latency_ms=25.0,
        prompt_tokens=120,
        completion_tokens=18,
    )
    summary = summarize_answer([case])
    assert case["generated"] is True
    assert case["grounded"] is True
    assert summary["generation_success_rate"] == 1.0
    assert summary["groundedness_rate"] == 1.0


def test_retrieval_case_uses_catalog() -> None:
    catalog = [
        ("c1", "Policy HR-203 grants annual leave", {"section_title": "Policy HR-203: Annual Leave"}),
        ("c2", "Policy HR-105 sick leave", {"section_title": "Policy HR-105: Sick Leave"}),
    ]
    case = evaluate_retrieval_case(
        case_id="R001",
        retrieved_ids=["c2", "c1"],
        catalog=catalog,
        gold_specs=[{"policy_id": "HR-203"}],
        expected_strategy="bm25",
        actual_strategy="bm25",
        top_k=2,
    )
    assert case["recall_at_k"] == 1.0
    assert case["mrr"] == 0.5


def test_dataset_files_exist() -> None:
    datasets_dir = Path("eval/datasets")
    expected = [
        "retrieval_dataset.jsonl",
        "rewrite_dataset.jsonl",
        "routing_dataset.jsonl",
        "decomposition_dataset.jsonl",
        "confidence_dataset.jsonl",
        "failure_dataset.jsonl",
        "answer_generation_dataset.jsonl",
        "citation_dataset.jsonl",
        "golden_demo.jsonl",
    ]
    for name in expected:
        assert (datasets_dir / name).exists()


@pytest.mark.integration
def test_run_eval_smoke(tmp_path: Path) -> None:
    from eval.run_eval import EvaluationRunner, write_report

    runner = EvaluationRunner(output_dir=tmp_path / "eval-run")
    report = runner.run_rewrite_eval()
    assert report["summary"]["count"] > 0

    full = runner.run_all()
    write_report(full, tmp_path / "report.json")
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.md").exists()
    payload = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert "phase_5_readiness" in payload
