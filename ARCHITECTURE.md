# Architecture

## Data Flow

```
                         ┌─────────────────────────────────────┐
                         │            Knowledge Base             │
                         │  historical_failures.json (12 records)│
                         │  defect_tickets.json (7 records)      │
                         │  code_commits.json (5 records)        │
                         │  runbooks/*.md (4 docs)                │
                         └───────────────────┬───────────────────┘
                                              │
                                     ingest.py (offline, run once
                                     or on knowledge-base update)
                                              │
                         ┌────────────────────▼───────────────────┐
                         │  1. Load records per source type         │
                         │  2. Chunk (see below)                    │
                         │  3. Fit TF-IDF vectorizer on full corpus │
                         │  4. Embed every chunk                    │
                         │  5. Persist: embeddings.npy,             │
                         │     documents.json, embedder.pkl         │
                         └────────────────────┬───────────────────┘
                                              │
                                     data/index/ (on disk)
                                              │
   new test failure ──► triage.py ──► retriever.py ──► vector_store.py
   (test name, msg,      build_query()   hybrid search:  cosine similarity
    stack trace)                         vector sim +     over persisted
                                          keyword boost    embeddings
                                              │
                                      top-k retrieved context
                                      (mixed source types)
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    │                                                     │
            ANTHROPIC_API_KEY set?                              no API key
                    │ yes                                                │ no
                    ▼                                                     ▼
        synthesize_with_llm()                          synthesize_extractive_fallback()
        - system prompt instructs                       - picks top match
          grounding + citation                           - formats evidence list
          discipline, no speculation                      - templates each report
        - Claude writes structured                          section from retrieved
          narrative report                                  metadata directly
                    │                                                     │
                    └─────────────────────────┬─────────────────────────┘
                                              ▼
                                   structured triage report
                                (Markdown, cited doc IDs, saved
                                 to outputs/ as JSON + printed)
```

## Chunking Strategy

Different source types get different chunking treatment because they have
different internal structure:

| Source | Chunk unit | Rationale |
|---|---|---|
| Historical failures | One chunk per record | Already atomic — the error message, root cause, and resolution are tightly coupled; splitting them would separate a symptom from its explanation. |
| Defect tickets | One chunk per ticket | Same reasoning — title, description, and resolution belong together. |
| Commits | One chunk per commit | A commit message + changed files is a single semantic unit relevant to "what changed recently." |
| Runbooks | One chunk per `## ` markdown section | Runbooks mix several distinct troubleshooting topics (e.g. one runbook covers 4-5 root-cause categories). Chunking at the section level means a query about "stale element exceptions" retrieves just that section, not an entire multi-topic document that dilutes relevance. |

This mirrors a general RAG principle: **chunk boundaries should match
semantic boundaries in the source material**, not an arbitrary fixed token
count. Fixed-size chunking is more defensible for large, uniformly-structured
prose (e.g. a legal contract) but actively hurts retrieval precision on
heterogeneous technical documents like these.

## Why Hybrid Retrieval

Pure vector similarity over TF-IDF (or any embedding model) can be
insufficiently sensitive to a small number of extremely high-value tokens:
an exact exception class name (`StaleElementReferenceException`), an HTTP
status code (`429`), or a commit SHA. Two documents can have very different
overall vocabulary but share one term that is, on its own, almost
deterministic of relevance.

`retriever.py` extracts these high-precision tokens via regex from both the
query and every candidate document, and adds a fixed boost per overlapping
token on top of the base cosine similarity score. This is a lightweight
version of the "sparse + dense hybrid retrieval" pattern used in production
RAG systems (e.g. combining BM25 with a dense retriever), scoped down to
fit a few hours of build time while still directly targeting the failure
mode it's meant to catch — see `eval/labeled_queries.json` cases involving
exact exception names and error codes.

## Why an Evaluation Harness Is Part of the Architecture, Not an Afterthought

From a QE perspective, an untested retrieval system is exactly the kind of
component that would fail an internal quality bar: it produces confident
looking answers with no verification that they're actually correct. The
`eval/` directory treats retrieval quality as a first-class testable
property (`recall@k` against a hand-labeled query set), not something
that's "probably fine because the demo looked good."

## Extending This Architecture

- **Continuous evaluation**: log every triage session's retrieved top match
  and whether a human confirmed or corrected it; append confirmed cases to
  `eval/labeled_queries.json` automatically to build a growing regression
  suite.
- **Freshness**: trigger `python -m src.ingest` from a CI job whenever
  `data/*.json` or `data/runbooks/*.md` change, or point the loaders at a
  live defect tracker / git log instead of static JSON files.
- **Feedback loop**: capture whether an engineer accepted or rejected the
  suggested root cause, and use it to tune the keyword-boost weight or
  eventually fine-tune a reranker.
