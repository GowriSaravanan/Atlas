"""Build the evaluation corpus PDF used by retrieval benchmarks."""

from __future__ import annotations

from pathlib import Path

import fitz

EVAL_COLLECTION_ID = "eval-corpus"


def build_eval_corpus_pdf(output_path: Path) -> Path:
    """Create a deterministic HR policy PDF for evaluation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    page = document.new_page()

    sections: list[tuple[str, int, str]] = [
        (
            "HR Leave Policy Handbook",
            18,
            "This handbook summarizes HR leave policies for full-time employees.",
        ),
        (
            "Policy HR-203: Annual Leave",
            16,
            "Policy HR-203 grants 20 days of annual leave per year for full-time HR employees.",
        ),
        (
            "Policy HR-105: Sick Leave",
            16,
            "Policy HR-105 allows up to 10 sick leave days per year with manager approval.",
        ),
        (
            "Maternity Leave",
            16,
            "Eligible employees receive 26 weeks of paid maternity leave under HR policy.",
        ),
        (
            "Paternity Leave",
            16,
            "Eligible employees receive 2 weeks of paid paternity leave under HR policy.",
        ),
    ]

    y = 72
    for title, size, body in sections:
        page.insert_text((72, y), title, fontsize=size)
        y += 28 if size >= 16 else 24
        page.insert_text((72, y), body, fontsize=11)
        y += 36

    document.save(output_path)
    document.close()
    return output_path
