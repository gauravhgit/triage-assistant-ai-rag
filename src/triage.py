"""
RAG triage pipeline: new failure -> retrieved context -> grounded report.

Two synthesis modes:
  1. LLM mode (default if ANTHROPIC_API_KEY is set): retrieved context is
     passed to Claude, which writes a structured triage report grounded in
     the retrieved evidence and instructed not to speculate beyond it.
  2. Extractive fallback (no API key required): the pipeline still runs
     end-to-end and produces a useful, if less fluent, report built directly
     from the top retrieved match and supporting evidence. This means
     anyone cloning the repo can run the full system with zero setup - a
     production RAG system should always degrade gracefully rather than
     hard-failing when a downstream LLM call isn't available.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.retriever import Retriever

TOP_K_DEFAULT = 5


@dataclass
class TriageInput:
    test_name: str
    suite: str
    failure_message: str
    stack_trace_snippet: str = ""


def build_query(failure: TriageInput) -> str:
    return (
        f"Test: {failure.test_name} (suite: {failure.suite})\n"
        f"Failure message: {failure.failure_message}\n"
        f"Stack trace: {failure.stack_trace_snippet}"
    )


def retrieve_context(retriever: Retriever, failure: TriageInput,
                      top_k: int = TOP_K_DEFAULT) -> List[Dict[str, Any]]:
    query = build_query(failure)
    return retriever.retrieve(query, top_k=top_k)


def _format_context_for_prompt(results: List[Dict[str, Any]]) -> str:
    blocks = []
    for r in results:
        blocks.append(
            f"[{r['source_type'].upper()} | id={r['doc_id']} | relevance={r['score']:.3f}]\n{r['text']}"
        )
    return "\n\n".join(blocks)


SYSTEM_PROMPT = """You are a senior QA/software quality engineer assisting with test \
failure triage. You will be given a new test failure and a set of retrieved \
context documents (past resolved failures, defect tickets, internal runbooks, \
and recent code commits). Using ONLY the provided context and the new failure \
details, produce a structured triage report with these sections:

1. Likely Root Cause Hypothesis (state your confidence: High/Medium/Low)
2. Supporting Evidence (cite specific retrieved document IDs for each point)
3. Is This Likely a New Issue or a Recurrence? (say which, and why)
4. Recommended Next Action (concrete, specific)
5. Suggested Owner/Component

If the retrieved context does not clearly support a conclusion, say so \
explicitly rather than guessing. Do not invent document IDs or facts not \
present in the context."""


def synthesize_with_llm(failure: TriageInput, context_results: List[Dict[str, Any]]) -> str:
    import anthropic  # imported lazily so the package is optional for fallback mode

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    context_block = _format_context_for_prompt(context_results)
    user_message = (
        f"NEW FAILURE:\n{build_query(failure)}\n\n"
        f"RETRIEVED CONTEXT:\n{context_block}\n\n"
        "Write the triage report now."
    )
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def synthesize_extractive_fallback(failure: TriageInput, context_results: List[Dict[str, Any]]) -> str:
    """
    No-LLM fallback: builds a structured report directly from retrieval
    results using simple heuristics. Runs with zero external dependencies
    or API keys.
    """
    if not context_results:
        return (
            "## Triage Report (extractive fallback - no ANTHROPIC_API_KEY set)\n\n"
            "**Likely Root Cause Hypothesis:** Unknown (Low confidence)\n\n"
            "No sufficiently similar historical context was found for this failure. "
            "This may be a genuinely new failure mode - recommend manual investigation."
        )

    top = context_results[0]
    historical_matches = [r for r in context_results if r["source_type"] == "historical_failure"]
    ticket_matches = [r for r in context_results if r["source_type"] == "defect_ticket"]
    runbook_matches = [r for r in context_results if r["source_type"] == "runbook"]
    commit_matches = [r for r in context_results if r["source_type"] == "commit"]

    confidence = "High" if top["score"] > 0.5 else ("Medium" if top["score"] > 0.25 else "Low")

    lines = ["## Triage Report (extractive fallback - no ANTHROPIC_API_KEY set)\n"]
    lines.append(f"**Likely Root Cause Hypothesis** (Confidence: {confidence}):")
    if top["source_type"] == "historical_failure":
        root_cause = top["metadata"].get("root_cause", "n/a")
        lines.append(f"> Based on closest historical match `{top['doc_id']}`: {root_cause}\n")
    else:
        lines.append(f"> Closest retrieved context is a {top['source_type']} (`{top['doc_id']}`); "
                      f"see evidence below.\n")

    lines.append("**Supporting Evidence:**")
    for r in context_results:
        tag = r["doc_id"]
        note = ""
        if r["matched_keywords"]:
            note = f" (matched terms: {', '.join(r['matched_keywords'])})"
        lines.append(f"- `{tag}` [{r['source_type']}, relevance={r['score']:.2f}]{note}")
    lines.append("")

    is_recurrence = bool(historical_matches) and historical_matches[0]["score"] > 0.4
    lines.append("**New Issue or Recurrence?**")
    if is_recurrence:
        lines.append(f"> Likely a **recurrence** of pattern seen in `{historical_matches[0]['doc_id']}`.\n")
    else:
        lines.append("> Insufficient similarity to a known historical failure to call this a "
                      "confirmed recurrence - treat as potentially **new** pending investigation.\n")

    lines.append("**Recommended Next Action:**")
    if top["source_type"] == "historical_failure":
        lines.append(f"> {top['metadata'].get('resolution', 'See linked resolution.')}")
    elif runbook_matches:
        lines.append(f"> Consult runbook `{runbook_matches[0]['doc_id']}` "
                      f"({runbook_matches[0]['metadata'].get('runbook_title', '')}) for diagnostic steps.")
    else:
        lines.append("> Manually investigate; no direct historical resolution found.")
    lines.append("")

    if commit_matches:
        lines.append("**Recent Related Changes:**")
        for c in commit_matches:
            lines.append(f"- `{c['doc_id']}`: {c['metadata'].get('message', '')} "
                          f"({c['metadata'].get('date', '')})")
        lines.append("")

    if ticket_matches:
        component = ticket_matches[0]["metadata"].get("component", "unknown")
        lines.append(f"**Suggested Owner/Component:** {component} team "
                      f"(based on related ticket `{ticket_matches[0]['doc_id']}`)")
    elif historical_matches:
        lines.append(f"**Suggested Owner/Component:** see `{historical_matches[0]['metadata'].get('resolved_by', 'n/a')}`, "
                      f"who resolved the closest similar past failure.")
    else:
        lines.append("**Suggested Owner/Component:** Unable to determine from retrieved context.")

    return "\n".join(lines)


def run_triage(failure: TriageInput, retriever: Optional[Retriever] = None,
               top_k: int = TOP_K_DEFAULT) -> Dict[str, Any]:
    if retriever is None:
        retriever = Retriever.from_disk()

    context_results = retrieve_context(retriever, failure, top_k=top_k)

    use_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if use_llm:
        try:
            report_text = synthesize_with_llm(failure, context_results)
            mode = "llm"
        except Exception as e:  # noqa: BLE001 - deliberate broad catch for graceful degradation
            report_text = (f"_LLM synthesis failed ({e}); falling back to extractive mode._\n\n"
                            + synthesize_extractive_fallback(failure, context_results))
            mode = "extractive_fallback_after_error"
    else:
        report_text = synthesize_extractive_fallback(failure, context_results)
        mode = "extractive_fallback"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "input_failure": failure.__dict__,
        "retrieved_context": context_results,
        "report": report_text,
    }
