"""Generate expanded evaluation datasets aligned with the eval corpus."""

from __future__ import annotations

import json
from pathlib import Path

DATASETS_DIR = Path(__file__).resolve().parents[1] / "datasets"


def write_jsonl(name: str, rows: list[dict]) -> None:
    path = DATASETS_DIR / name
    path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")


def main() -> None:
    routing = [
        {"id": "RT001", "query": "What is HR-203?", "expected_strategy": "bm25"},
        {"id": "RT002", "query": "What is HR-105?", "expected_strategy": "bm25"},
        {"id": "RT003", "query": "Lookup policy HR-203", "expected_strategy": "bm25"},
        {"id": "RT004", "query": "Find policy identifier HR-105", "expected_strategy": "bm25"},
        {"id": "RT005", "query": "Explain employee maternity leave benefits in detail", "expected_strategy": "hybrid"},
        {"id": "RT006", "query": "How many sick leave days are allowed?", "expected_strategy": "hybrid"},
        {"id": "RT007", "query": "What is the annual leave policy for new hires?", "expected_strategy": "hybrid"},
        {"id": "RT008", "query": "Compare HR-203 with HR-105", "expected_strategy": "hybrid"},
        {"id": "RT009", "query": "Compare maternity leave and paternity leave benefits.", "expected_strategy": "hybrid"},
        {"id": "RT010", "query": "Difference between HR-203 and HR-105", "expected_strategy": "hybrid"},
        {"id": "RT011", "query": "annual leave 20 days HR department", "expected_strategy": "hybrid"},
        {"id": "RT012", "query": "Summarize HR leave policies for employees", "expected_strategy": "hybrid"},
        {"id": "RT013", "query": "How do I apply for sick leave?", "expected_strategy": "hybrid"},
        {"id": "RT014", "query": "Explain paid parental leave options", "expected_strategy": "hybrid"},
        {"id": "RT015", "query": "What leave benefits exist for parents?", "expected_strategy": "hybrid"},
        {"id": "RT016", "query": "paternity leave weeks", "expected_strategy": "dense"},
        {"id": "RT017", "query": "maternity leave eligibility", "expected_strategy": "dense"},
        {"id": "RT018", "query": "sick leave days", "expected_strategy": "dense"},
        {"id": "RT019", "query": "annual leave policy", "expected_strategy": "hybrid"},
        {"id": "RT020", "query": "Compare maternity leave with paternity leave", "expected_strategy": "hybrid"},
        {"id": "RT021", "query": "What is the sick leave policy?", "expected_strategy": "hybrid"},
        {"id": "RT022", "query": "HR handbook leave summary", "expected_strategy": "hybrid"},
        {"id": "RT023", "query": "Policy HR-203 annual leave days", "expected_strategy": "bm25"},
        {"id": "RT024", "query": "Versus HR-203 and HR-105 leave policies", "expected_strategy": "hybrid"},
        {"id": "RT025", "query": "Tell me about paternity leave policy", "expected_strategy": "hybrid"},
    ]

    retrieval = [
        {"id": "R001", "query": "What is HR-203?", "expected_strategy": "bm25", "gold": [{"policy_id": "HR-203"}], "top_k": 5},
        {"id": "R002", "query": "Explain employee maternity leave benefits", "expected_strategy": "hybrid", "gold": [{"section_title": "Maternity Leave"}], "top_k": 5},
        {"id": "R003", "query": "How many sick leave days are allowed?", "expected_strategy": "hybrid", "gold": [{"policy_id": "HR-105"}], "top_k": 5},
        {"id": "R004", "query": "annual leave 20 days HR department", "expected_strategy": "hybrid", "gold": [{"policy_id": "HR-203"}], "top_k": 5},
        {"id": "R005", "query": "paternity leave weeks", "expected_strategy": "dense", "gold": [{"section_title": "Paternity Leave"}], "top_k": 5},
        {"id": "R006", "query": "What is HR-105?", "expected_strategy": "bm25", "gold": [{"policy_id": "HR-105"}], "top_k": 5},
        {"id": "R007", "query": "26 weeks maternity leave", "expected_strategy": "hybrid", "gold": [{"section_title": "Maternity Leave"}], "top_k": 5},
        {"id": "R008", "query": "2 weeks paternity leave", "expected_strategy": "hybrid", "gold": [{"section_title": "Paternity Leave"}], "top_k": 5},
        {"id": "R009", "query": "Policy HR-203 annual leave entitlement", "expected_strategy": "bm25", "gold": [{"policy_id": "HR-203"}], "top_k": 5},
        {"id": "R010", "query": "10 sick leave days manager approval", "expected_strategy": "hybrid", "gold": [{"policy_id": "HR-105"}], "top_k": 5},
        {"id": "R011", "query": "HR leave policy handbook summary", "expected_strategy": "hybrid", "gold": [{"section_title": "HR Leave Policy Handbook"}], "top_k": 5},
        {"id": "R012", "query": "full-time employees annual leave days", "expected_strategy": "hybrid", "gold": [{"policy_id": "HR-203"}], "top_k": 5},
        {"id": "R013", "query": "paid maternity leave HR policy", "expected_strategy": "hybrid", "gold": [{"section_title": "Maternity Leave"}], "top_k": 5},
        {"id": "R014", "query": "paid paternity leave HR policy", "expected_strategy": "hybrid", "gold": [{"section_title": "Paternity Leave"}], "top_k": 5},
        {"id": "R015", "query": "Lookup HR-203 policy document", "expected_strategy": "bm25", "gold": [{"policy_id": "HR-203"}], "top_k": 5},
        {"id": "R016", "query": "sick leave policy HR-105", "expected_strategy": "hybrid", "gold": [{"policy_id": "HR-105"}], "top_k": 5},
        {"id": "R017", "query": "Compare maternity and paternity leave benefits", "expected_strategy": "hybrid", "gold": [{"section_title": "Maternity Leave"}, {"section_title": "Paternity Leave"}], "top_k": 5},
        {"id": "R018", "query": "20 days annual leave full-time", "expected_strategy": "hybrid", "gold": [{"policy_id": "HR-203"}], "top_k": 5},
        {"id": "R019", "query": "employee handbook leave policies", "expected_strategy": "hybrid", "gold": [{"section_title": "HR Leave Policy Handbook"}], "top_k": 5},
        {"id": "R020", "query": "What is the sick leave policy?", "expected_strategy": "hybrid", "gold": [{"policy_id": "HR-105"}], "top_k": 5},
        {"id": "R021", "query": "What is the annual leave policy?", "expected_strategy": "hybrid", "gold": [{"policy_id": "HR-203"}], "top_k": 5},
        {"id": "R022", "query": "maternity leave policy details", "expected_strategy": "hybrid", "gold": [{"section_title": "Maternity Leave"}], "top_k": 5},
        {"id": "R023", "query": "paternity leave policy details", "expected_strategy": "hybrid", "gold": [{"section_title": "Paternity Leave"}], "top_k": 5},
        {"id": "R024", "query": "HR-105 sick days per year", "expected_strategy": "bm25", "gold": [{"policy_id": "HR-105"}], "top_k": 5},
        {"id": "R025", "query": "HR-203 leave entitlement", "expected_strategy": "bm25", "gold": [{"policy_id": "HR-203"}], "top_k": 5},
    ]

    rewrite = [
        {"id": "RW001", "query": "What about maternity leave?", "expected_rewrite": "What is the maternity leave policy?", "rewrite_type": "EXPANSION"},
        {"id": "RW002", "query": "How about paternity leave?", "expected_rewrite": "What is the paternity leave policy?", "rewrite_type": "EXPANSION"},
        {"id": "RW003", "query": "What is HR-203?", "expected_rewrite": "What is HR-203?", "rewrite_type": "PASS_THROUGH"},
        {"id": "RW004", "query": "Tell me about sick leave", "expected_rewrite": "What is the sick leave policy?", "rewrite_type": "EXPANSION"},
        {"id": "RW005", "query": "What about annual leave?", "expected_rewrite": "What is the annual leave policy?", "rewrite_type": "EXPANSION"},
        {"id": "RW006", "query": "How about HR-105?", "expected_rewrite": "What is the HR-105 policy?", "rewrite_type": "EXPANSION"},
        {"id": "RW007", "query": "Tell me about maternity leave policy", "expected_rewrite": "What is maternity leave policy?", "rewrite_type": "EXPANSION"},
        {"id": "RW008", "query": "What about paternity leave benefits?", "expected_rewrite": "What is the paternity leave benefits policy?", "rewrite_type": "EXPANSION"},
        {"id": "RW009", "query": "What is the sick leave policy?", "expected_rewrite": "What is the sick leave policy?", "rewrite_type": "PASS_THROUGH"},
        {"id": "RW010", "query": "What is HR-105?", "expected_rewrite": "What is HR-105?", "rewrite_type": "PASS_THROUGH"},
        {"id": "RW011", "query": "Tell me about annual leave policy", "expected_rewrite": "What is annual leave policy?", "rewrite_type": "EXPANSION"},
        {"id": "RW012", "query": "How about sick leave days?", "expected_rewrite": "What is the sick leave days policy?", "rewrite_type": "EXPANSION"},
        {"id": "RW013", "query": "What about HR-203?", "expected_rewrite": "What is the HR-203 policy?", "rewrite_type": "EXPANSION"},
        {"id": "RW014", "query": "Tell me about paternity leave", "expected_rewrite": "What is the paternity leave policy?", "rewrite_type": "EXPANSION"},
        {"id": "RW015", "query": "Explain annual leave benefits", "expected_rewrite": "Explain annual leave benefits", "rewrite_type": "PASS_THROUGH"},
        {"id": "RW016", "query": "What about leave carry over?", "expected_rewrite": "What is the leave carry over policy?", "rewrite_type": "EXPANSION"},
        {"id": "RW017", "query": "How about maternity benefits?", "expected_rewrite": "What is the maternity benefits policy?", "rewrite_type": "EXPANSION"},
        {"id": "RW018", "query": "What is the maternity leave policy?", "expected_rewrite": "What is the maternity leave policy?", "rewrite_type": "PASS_THROUGH"},
        {"id": "RW019", "query": "Tell me about HR leave handbook", "expected_rewrite": "What is HR leave handbook?", "rewrite_type": "EXPANSION"},
        {"id": "RW020", "query": "What about it?", "expected_rewrite": "What about it?", "rewrite_type": "AMBIGUOUS_NO_CONTEXT"},
    ]

    decomposition = [
        {"id": "D001", "query": "Compare maternity leave and paternity leave.", "should_decompose": True, "expected_subqueries": ["What is the maternity leave policy?", "What is the paternity leave policy?"]},
        {"id": "D002", "query": "Compare maternity leave and paternity leave benefits.", "should_decompose": True, "expected_subqueries": ["What is the maternity leave policy?", "What is the paternity leave policy?"]},
        {"id": "D003", "query": "What is the annual leave policy and how many days can I carry over?", "should_decompose": False},
        {"id": "D004", "query": "What is HR-203?", "should_decompose": False},
        {"id": "D005", "query": "Difference between HR-203 and HR-105", "should_decompose": True, "expected_subqueries": ["What is HR-203?", "What is HR-105?"]},
        {"id": "D006", "query": "Compare HR-203 with HR-105", "should_decompose": True, "expected_subqueries": ["What is HR-203?", "What is HR-105?"]},
        {"id": "D007", "query": "Compare maternity leave with paternity leave", "should_decompose": True, "expected_subqueries": ["What is the maternity leave policy?", "What is the paternity leave policy?"]},
        {"id": "D008", "query": "What is the sick leave policy?", "should_decompose": False},
        {"id": "D009", "query": "Explain maternity leave and paternity leave benefits", "should_decompose": False},
        {"id": "D010", "query": "Versus HR-203 and HR-105 leave policies", "should_decompose": True, "expected_subqueries": ["What is HR-203?", "What is HR-105?"]},
        {"id": "D011", "query": "How many sick leave days and annual leave days?", "should_decompose": False},
        {"id": "D012", "query": "What is HR-105?", "should_decompose": False},
        {"id": "D013", "query": "Compare annual leave and sick leave policies", "should_decompose": True, "expected_subqueries": ["What is the annual leave policy?", "What is the sick leave policy?"]},
        {"id": "D014", "query": "What is the annual leave policy for 2024?", "should_decompose": False},
        {"id": "D015", "query": "Tell me about maternity leave", "should_decompose": False},
        {"id": "D016", "query": "Compare HR-203 and HR-105 benefits", "should_decompose": True, "expected_subqueries": ["What is HR-203?", "What is HR-105?"]},
        {"id": "D017", "query": "Annual leave policy and carry over rules", "should_decompose": False},
        {"id": "D018", "query": "Difference between maternity leave and paternity leave benefits", "should_decompose": True, "expected_subqueries": ["What is the maternity leave policy?", "What is the paternity leave policy?"]},
        {"id": "D019", "query": "What is the paternity leave policy?", "should_decompose": False},
        {"id": "D020", "query": "Compare sick leave and annual leave for new hires", "should_decompose": True, "expected_subqueries": ["What is the sick leave policy?", "What is the annual leave policy?"]},
    ]

    confidence = [
        {"id": "C001", "query": "What is HR-203?", "expected_confidence": "high", "gold": [{"policy_id": "HR-203"}]},
        {"id": "C002", "query": "Explain maternity leave eligibility", "expected_confidence": "high", "gold": [{"section_title": "Maternity Leave"}]},
        {"id": "C003", "query": "What is the vacation reimbursement policy in 2035?", "expected_confidence": "low", "gold": [{"content_contains": "vacation reimbursement 2035"}]},
        {"id": "C004", "query": "Tell me everything about HR.", "expected_confidence": "medium", "gold": [{"content_contains": "__missing__"}]},
        {"id": "C005", "query": "What is HR-105?", "expected_confidence": "high", "gold": [{"policy_id": "HR-105"}]},
        {"id": "C006", "query": "How many sick leave days?", "expected_confidence": "high", "gold": [{"policy_id": "HR-105"}]},
        {"id": "C007", "query": "paternity leave weeks", "expected_confidence": "medium", "gold": [{"section_title": "Paternity Leave"}]},
        {"id": "C008", "query": "What is the Mars relocation policy?", "expected_confidence": "low", "gold": [{"content_contains": "Mars relocation"}]},
        {"id": "C009", "query": "annual leave 20 days", "expected_confidence": "high", "gold": [{"policy_id": "HR-203"}]},
        {"id": "C010", "query": "26 weeks maternity leave", "expected_confidence": "high", "gold": [{"section_title": "Maternity Leave"}]},
        {"id": "C011", "query": "What is the unlimited vacation policy?", "expected_confidence": "low", "gold": [{"content_contains": "unlimited vacation"}]},
        {"id": "C012", "query": "HR leave handbook summary", "expected_confidence": "medium", "gold": [{"section_title": "HR Leave Policy Handbook"}]},
        {"id": "C013", "query": "2 weeks paternity leave", "expected_confidence": "high", "gold": [{"section_title": "Paternity Leave"}]},
        {"id": "C014", "query": "What is policy XYZ-999?", "expected_confidence": "low", "gold": [{"policy_id": "XYZ-999"}]},
        {"id": "C015", "query": "Explain employee leave benefits", "expected_confidence": "medium", "gold": [{"section_title": "HR Leave Policy Handbook"}]},
        {"id": "C016", "query": "Policy HR-203 entitlement", "expected_confidence": "high", "gold": [{"policy_id": "HR-203"}]},
        {"id": "C017", "query": "What is the sabbatical policy?", "expected_confidence": "low", "gold": [{"content_contains": "sabbatical"}]},
        {"id": "C018", "query": "sick leave manager approval", "expected_confidence": "high", "gold": [{"policy_id": "HR-105"}]},
        {"id": "C019", "query": "Tell me everything.", "expected_confidence": "low", "gold": [{"content_contains": "__missing__"}]},
        {"id": "C020", "query": "What is the annual leave policy?", "expected_confidence": "high", "gold": [{"policy_id": "HR-203"}]},
    ]

    failure = [
        {"id": "F001", "query": "What is the vacation reimbursement policy in 2035?", "expected_behavior": "no_evidence", "gold": [{"content_contains": "vacation reimbursement 2035"}]},
        {"id": "F002", "query": "Tell me everything about HR policies.", "expected_behavior": "low_confidence", "gold": [{"content_contains": "__missing__"}]},
        {"id": "F003", "query": "What about it?", "expected_behavior": "rewrite_or_clarify", "context_messages": [{"role": "user", "content": "Tell me about maternity leave policy"}]},
        {"id": "F004", "query": "What is the annual leave policy and how many days can I carry over?", "expected_behavior": "no_false_decomposition"},
        {"id": "F005", "query": "What is the Mars colony leave policy?", "expected_behavior": "no_evidence", "gold": [{"content_contains": "Mars colony"}]},
        {"id": "F006", "query": "Tell me everything.", "expected_behavior": "rewrite_or_clarify"},
        {"id": "F007", "query": "What is policy XYZ-999?", "expected_behavior": "no_evidence", "gold": [{"policy_id": "XYZ-999"}]},
        {"id": "F008", "query": "Compare vacation and sabbatical leave globally", "expected_behavior": "no_evidence", "gold": [{"content_contains": "sabbatical"}]},
        {"id": "F009", "query": "What about leave?", "expected_behavior": "rewrite_or_clarify"},
        {"id": "F010", "query": "What is the unlimited PTO policy?", "expected_behavior": "no_evidence", "gold": [{"content_contains": "unlimited PTO"}]},
        {"id": "F011", "query": "Explain every HR policy in detail", "expected_behavior": "low_confidence"},
        {"id": "F012", "query": "What is the 2035 parental leave reform?", "expected_behavior": "no_evidence", "gold": [{"content_contains": "2035 parental leave reform"}]},
        {"id": "F013", "query": "What about it?", "expected_behavior": "rewrite_or_clarify"},
        {"id": "F014", "query": "Annual leave policy and carry over days", "expected_behavior": "no_false_decomposition"},
        {"id": "F015", "query": "What is the dental coverage policy?", "expected_behavior": "no_evidence", "gold": [{"content_contains": "dental coverage"}]},
        {"id": "F016", "query": "How about that?", "expected_behavior": "rewrite_or_clarify", "context_messages": [{"role": "user", "content": "What is HR-203?"}]},
        {"id": "F017", "query": "Compare HR-999 with HR-888", "expected_behavior": "no_evidence"},
        {"id": "F018", "query": "What is the remote work policy in Antarctica?", "expected_behavior": "no_evidence", "gold": [{"content_contains": "Antarctica"}]},
        {"id": "F019", "query": "Hello", "expected_behavior": "rewrite_or_clarify"},
        {"id": "F020", "query": "What is the stock option vesting cliff?", "expected_behavior": "no_evidence", "gold": [{"content_contains": "stock option vesting"}]},
    ]

    write_jsonl("routing_dataset.jsonl", routing)
    write_jsonl("retrieval_dataset.jsonl", retrieval)
    write_jsonl("rewrite_dataset.jsonl", rewrite)
    write_jsonl("decomposition_dataset.jsonl", decomposition)
    write_jsonl("confidence_dataset.jsonl", confidence)
    write_jsonl("failure_dataset.jsonl", failure)
    print("Wrote expanded datasets to", DATASETS_DIR)


if __name__ == "__main__":
    main()
