"""
CLI for the RAG test-failure triage assistant.

Usage:
    python -m src.cli --failure examples/new_failure_example.json
    python -m src.cli --failure examples/new_failure_example.json --top-k 3 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.retriever import Retriever
from src.triage import TriageInput, run_triage

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def main():
    parser = argparse.ArgumentParser(description="RAG-powered test failure triage")
    parser.add_argument("--failure", required=True, help="Path to a new-failure JSON file")
    parser.add_argument("--top-k", type=int, default=5, help="Number of context docs to retrieve")
    parser.add_argument("--json", action="store_true", help="Print full JSON result instead of just the report")
    parser.add_argument("--no-save", action="store_true", help="Don't write a report file to outputs/")
    args = parser.parse_args()

    failure_path = Path(args.failure)
    if not failure_path.exists():
        print(f"Error: {failure_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(failure_path) as f:
        raw = json.load(f)
    failure = TriageInput(
        test_name=raw["test_name"],
        suite=raw.get("suite", "unknown"),
        failure_message=raw["failure_message"],
        stack_trace_snippet=raw.get("stack_trace_snippet", ""),
    )

    try:
        retriever = Retriever.from_disk()
    except FileNotFoundError:
        print("Index not found. Build it first with:\n    python -m src.ingest", file=sys.stderr)
        sys.exit(1)

    result = run_triage(failure, retriever=retriever, top_k=args.top_k)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(result["report"])
        print(f"\n---\n(synthesis mode: {result['mode']})")

    if not args.no_save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"triage_{failure.test_name}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nFull result saved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
