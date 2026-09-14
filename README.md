# RAG-Powered Test Failure Triage Assistant

A Retrieval-Augmented Generation (RAG) system that grounds test failure triage in
an org's own historical data such as test failures, defect tickets, runbooks, and
recent commits instead of relying solely on an LLM's general knowledge (pre-training 
data). Without AI this task was generally performed through manual searches and human
memory and intuition. Tools such as this allow LLMs to augment or automate that process.

This is intended as a portfolio project demonstrating application of AI/ML in software
quality engineering. All data (test failures, tickets, runbooks, commits) is synthetic, 
generated to resemble realistic QA organization artifacts.

## The Problem

When a test fails in CI, a QA engineer has to answer, from scratch, every
time: *Is this a known flaky test? A real regression? Caused by a recent
code change? An environment issue? Something the team already fixed six
months ago and forgot?*

That institutional memory usually lives scattered across a defect tracker,
a wiki, Slack threads, and people's heads. This project shows how RAG can
surface it automatically at the moment a test fails.

## What It Does

Given a new test failure (test name, failure message, stack trace), the
system:

1. **Retrieves** the most relevant historical failures, defect tickets,
   runbook sections, and recent commits using hybrid (vector + keyword)
   search over a knowledge base built from synthetic QA artifacts.
2. **Synthesizes** a structured triage report — root cause hypothesis with
   a confidence level, cited supporting evidence, a new-vs-recurring-issue
   call, a recommended next action, and a suggested owner/component.
3. **Grounds every claim in retrieved evidence** rather than letting the
   model guess — the prompt explicitly instructs it not to speculate beyond
   the provided context, and every report cites specific document IDs.
