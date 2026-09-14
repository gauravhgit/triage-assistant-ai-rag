"""
Unit tests for retrieval and the extractive triage fallback.

Assumes the index has already been built (python -m src.ingest). CI should
run ingestion before tests - see README.md "Running Tests".
"""

import pytest

from src.retriever import Retriever
from src.triage import TriageInput, run_triage


@pytest.fixture(scope="module")
def retriever():
    try:
        return Retriever.from_disk()
    except FileNotFoundError:
        pytest.skip("Index not built - run `python -m src.ingest` first.")


def test_retrieve_returns_results(retriever):
    results = retriever.retrieve("read timeout payment gateway", top_k=5)
    assert len(results) > 0
    assert len(results) <= 5


def test_retrieve_top_k_respected(retriever):
    results = retriever.retrieve("assertion error currency", top_k=2)
    assert len(results) <= 2


def test_exact_error_match_ranks_highly(retriever):
    """A query containing an exact exception class name should retrieve
    the historical failure that shares that exact class name near the top."""
    results = retriever.retrieve(
        "StaleElementReferenceException while running checkout e2e test", top_k=5
    )
    top_ids = [r["doc_id"] for r in results]
    assert "HF-002" in top_ids


def test_source_type_filter(retriever):
    results = retriever.retrieve("timeout", top_k=10, source_types=["runbook"])
    assert all(r["source_type"] == "runbook" for r in results)


def test_empty_index_query_type_returns_empty(retriever):
    results = retriever.retrieve("timeout", top_k=5, source_types=["nonexistent_type"])
    assert results == []


def test_run_triage_extractive_fallback_end_to_end(retriever, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    failure = TriageInput(
        test_name="test_new_payment_flow",
        suite="payments-integration",
        failure_message="ReadTimeoutError: HTTPSConnectionPool(host='api.paymentsvc.internal', port=443): Read timed out.",
        stack_trace_snippet="requests.exceptions.ReadTimeout",
    )
    result = run_triage(failure, retriever=retriever, top_k=5)
    assert result["mode"] == "extractive_fallback"
    assert "Root Cause Hypothesis" in result["report"]
    assert len(result["retrieved_context"]) > 0


def test_run_triage_handles_no_relevant_context(retriever):
    failure = TriageInput(
        test_name="test_totally_unrelated_thing",
        suite="unknown-suite",
        failure_message="zzz qux flibbertigibbet nonsense string unrelated to anything",
    )
    result = run_triage(failure, retriever=retriever, top_k=5)
    assert "report" in result
