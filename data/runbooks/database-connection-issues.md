# Runbook: Database Connection Timeout / Pool Exhaustion

## Symptoms
- `OperationalError: could not connect to server: Connection timed out`
- Failures cluster later in a test suite run, not at the start
- Failure rate drops to zero after CI worker/process restart

## Likely Causes
1. Connection pool exhaustion from tests that open a DB connection but do
   not close it in a `finally` block or context manager, especially on the
   failure path of a test (exception raised before cleanup runs).
2. Genuine DB/network issue in the target environment (check DB server
   metrics and network health before assuming a test-code bug).
3. Pool size configured too small for the level of test parallelism in use.

## Diagnostic Steps
1. Check whether failures start only after N tests have run in the suite -
   if so, suspect a leak, not an environment outage.
2. Grep the suite for direct `connection = get_connection()` calls not
   wrapped in `with` or paired with an explicit `finally: connection.close()`.
3. Compare pool size config against `pytest-xdist` worker count.

## Standard Fix
Use context-manager based connection handling everywhere, and add a teardown
assertion (e.g. in a session-scoped fixture) that the pool's active
connection count returns to its baseline after each test. This turns future
leaks into an immediate, localized test failure instead of a mysterious
failure 40 tests later.
