"""
Ingestion pipeline: load raw QE artifacts -> chunk -> embed -> persist index.

Run as a script to (re)build the index whenever source data changes:
    python -m src.ingest

Chunking strategy (see ARCHITECTURE.md for rationale):
- historical_failures / defect_tickets / commits: one chunk per record.
  These are already small, semantically atomic units - splitting them
  further would separate an error message from its root cause.
- runbooks: split by markdown "## " section headers. A runbook mixes
  several distinct troubleshooting topics; section-level chunks keep
  retrieval precise instead of returning an entire multi-topic document.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List

from src.embeddings import get_default_embedder
from src.vector_store import Document, VectorStore

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "index"


def load_historical_failures() -> List[Document]:
    with open(DATA_DIR / "historical_failures.json") as f:
        records = json.load(f)
    docs = []
    for r in records:
        text = (
            f"Test: {r['test_name']} (suite: {r['suite']})\n"
            f"Failure message: {r['failure_message']}\n"
            f"Stack trace: {r['stack_trace_snippet']}\n"
            f"Root cause: {r['root_cause']}\n"
            f"Resolution: {r['resolution']}\n"
            f"Tags: {', '.join(r['tags'])}"
        )
        docs.append(Document(
            doc_id=r["id"], text=text, source_type="historical_failure",
            metadata=r,
        ))
    return docs


def load_defect_tickets() -> List[Document]:
    with open(DATA_DIR / "defect_tickets.json") as f:
        records = json.load(f)
    docs = []
    for r in records:
        text = (
            f"Ticket {r['id']} [{r['component']}] - {r['title']}\n"
            f"Description: {r['description']}\n"
            f"Resolution: {r['resolution']}\n"
            f"Status: {r['status']}"
        )
        docs.append(Document(
            doc_id=r["id"], text=text, source_type="defect_ticket",
            metadata=r,
        ))
    return docs


def load_commits() -> List[Document]:
    with open(DATA_DIR / "code_commits.json") as f:
        records = json.load(f)
    docs = []
    for r in records:
        text = (
            f"Commit {r['sha']} ({r['date']}, {r['component']}): {r['message']}\n"
            f"Files changed: {', '.join(r['files_changed'])}"
        )
        docs.append(Document(
            doc_id=r["sha"], text=text, source_type="commit", metadata=r,
        ))
    return docs


def load_runbooks() -> List[Document]:
    docs = []
    runbook_dir = DATA_DIR / "runbooks"
    for path in sorted(runbook_dir.glob("*.md")):
        content = path.read_text()
        title_line = content.splitlines()[0].lstrip("# ").strip()
        sections = _split_markdown_sections(content)
        for i, (heading, body) in enumerate(sections):
            doc_id = f"{path.stem}::{i}"
            text = f"[Runbook: {title_line} - {heading}]\n{body}"
            docs.append(Document(
                doc_id=doc_id, text=text, source_type="runbook",
                metadata={"file": path.name, "section": heading, "runbook_title": title_line},
            ))
    return docs


def _split_markdown_sections(content: str):
    """Split a markdown doc into (heading, body) chunks on '## ' headers."""
    lines = content.splitlines()
    sections = []
    current_heading = "Overview"
    current_body: List[str] = []
    for line in lines[1:]:  # skip the H1 title
        if line.startswith("## "):
            if current_body:
                sections.append((current_heading, "\n".join(current_body).strip()))
            current_heading = line.lstrip("# ").strip()
            current_body = []
        else:
            current_body.append(line)
    if current_body:
        sections.append((current_heading, "\n".join(current_body).strip()))
    return [(h, b) for h, b in sections if b]


def build_index() -> VectorStore:
    all_docs: List[Document] = []
    all_docs += load_historical_failures()
    all_docs += load_defect_tickets()
    all_docs += load_commits()
    all_docs += load_runbooks()

    embedder = get_default_embedder()
    texts = [d.text for d in all_docs]
    embedder.fit(texts)
    embeddings = embedder.embed(texts)

    store = VectorStore()
    store.add(all_docs, embeddings)
    return store, embedder


if __name__ == "__main__":
    store, embedder = build_index()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save(str(INDEX_DIR))
    with open(INDEX_DIR / "embedder.pkl", "wb") as f:
        pickle.dump(embedder, f)
    print(f"Indexed {len(store.documents)} documents "
          f"({sum(1 for d in store.documents if d.source_type == 'historical_failure')} historical failures, "
          f"{sum(1 for d in store.documents if d.source_type == 'defect_ticket')} tickets, "
          f"{sum(1 for d in store.documents if d.source_type == 'commit')} commits, "
          f"{sum(1 for d in store.documents if d.source_type == 'runbook')} runbook sections)")
    print(f"Index saved to {INDEX_DIR}")
