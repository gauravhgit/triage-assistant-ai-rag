# Runbook: Diagnosing Flaky Tests

## Definition
A test is considered flaky if it produces both pass and fail results across
multiple runs against the same code revision with no code changes in between.

## Common Root Causes
1. **Test isolation failures** - shared state (DB connections, global
   variables, files) leaking between tests. Symptom: failure rate increases
   the longer a suite runs, or failures cluster after a specific test.
2. **Timing / async race conditions** - assertions run before an async
   operation (network call, UI render, background job) completes. Symptom:
   intermittent timeouts or "element not found" style errors.
3. **External dependency instability** - third-party sandboxes/APIs with
   lower reliability or rate limits than production. Symptom: failures
   correlate with specific vendors and with parallel/high-concurrency runs.
4. **Environment drift** - CI runner image changes, missing system
   dependencies, or version mismatches. Symptom: failures start suddenly
   across many unrelated tests at the same time, tracing back to an infra change.

## Triage Checklist
- Does the test fail consistently or intermittently on re-run?
- Did a recent commit touch the code path under test?
- Did a recent CI/infra change occur around the failure's first occurrence?
- Is the failure isolated to one test or does it correlate with parallel execution / worker count?
- Is there an external dependency (API, email, SMS, third-party sandbox) involved?

## Recommended Fixes by Category
| Category | Fix pattern |
|---|---|
| Test isolation | Use fixtures with explicit teardown; assert resource counts return to baseline after each test |
| Timing/race | Replace fixed sleeps with explicit polling/wait conditions on the actual signal (not time) |
| External dependency | Add retry-with-backoff, increase timeout defensively, or add a rate limiter fixture |
| Environment drift | Pin dependency versions explicitly in CI config rather than relying on base image defaults |
