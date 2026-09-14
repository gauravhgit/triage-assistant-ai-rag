# Runbook: Third-Party API Rate Limiting in Tests

## Symptoms
- HTTP 429 responses from an external vendor sandbox
- Failure rate scales with the number of parallel test workers
- Tests pass reliably when run serially but fail under `pytest-xdist`

## Root Cause Pattern
Vendor sandboxes almost always have lower rate limits than production.
Parallelizing a test suite (common for speeding up CI) multiplies the
effective request rate against a shared sandbox account, exceeding the
vendor's limit even though no single test is doing anything wrong.

## Standard Fix
Introduce a session-scoped rate limiter (e.g. a token-bucket fixture) shared
across all tests that call the vendor sandbox, so the effective request rate
respects the vendor's documented limit regardless of worker count. Avoid
simply adding retries alone - retries under sustained overload just delay
the same failure and slow down CI.

## When This Is NOT the Cause
If failures also occur when running fully serially (1 worker), suspect
vendor-side sandbox instability instead and escalate to the vendor rather
than tuning test concurrency.
