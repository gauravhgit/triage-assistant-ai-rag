"""
Retrieval evaluation harness.

A RAG system is, itself, a piece of software that needs testing. This
script applies a basic IR evaluation methodology (recall@k) against a small
hand-labeled set of (query, expected_relevant_doc_ids) pairs, so retrieval
quality is measured rather than assumed. In a real project this labeled set
would grow over time from real triage sessions where a human confirmed
(or corrected) the top match.

Run:
    python -m eval.eval_retrieval
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from src.retriever import Retriever

LABELED_SET_PATH = Path(__file__).resolve().parent / "labeled_queries.json"


def recall_at_k(retrieved_ids: List[str], expected_ids: List[str]) -> float:
    if not expected_ids:
        return 1.0
    hit = any(doc_id in retrieved_ids for doc_id in expected_ids)
    return 1.0 if hit else 0.0


def main():
    retriever = Retriever.from_disk()
    with open(LABELED_SET_PATH) as f:
        cases = json.load(f)

    results = []
    for case in cases:
        query = case["query"]
        expected = case["expected_relevant_doc_ids"]
        top_k = case.get("top_k", 5)

        retrieved = retriever.retrieve(query, top_k=top_k)
        retrieved_ids = [r["doc_id"] for r in retrieved]
        r_at_k = recall_at_k(retrieved_ids, expected)

        results.append({
            "case": case["name"],
            "recall_at_k": r_at_k,
            "retrieved": retrieved_ids,
            "expected": expected,
        })

    passed = sum(1 for r in results if r["recall_at_k"] == 1.0)
    print(f"Recall@k: {passed}/{len(results)} cases passed\n")
    for r in results:
        status = "PASS" if r["recall_at_k"] == 1.0 else "FAIL"
        print(f"[{status}] {r['case']}")
        print(f"       expected one of: {r['expected']}")
        print(f"       retrieved:       {r['retrieved']}")
    print()

    if passed < len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
