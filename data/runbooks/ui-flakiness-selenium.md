# Runbook: Selenium/UI Test Flakiness

## Common Exceptions and What They Mean
- **StaleElementReferenceException**: the DOM node behind a cached
  `WebElement` was removed/replaced (e.g. React/Vue re-render) after the
  reference was captured but before it was used.
- **TimeoutException waiting for element**: the element never appeared
  within the wait window - could be a real UI bug, or the wait duration is
  too short for an async dependency (API call, email delivery, etc.)
  feeding that element.

## Anti-Patterns to Flag in Code Review
- Caching a `WebElement` reference across multiple actions instead of
  re-locating it right before each use.
- Fixed `time.sleep(n)` instead of explicit wait conditions.
- Wait timeouts shorter than the known latency of an async dependency
  (check runbooks for that dependency, e.g. email/SMS delivery times).

## Standard Fixes
- Re-locate elements immediately before interacting with them, inside an
  explicit wait condition (`WebDriverWait(...).until(...)`), rather than
  reusing a captured reference.
- For async dependencies with unpredictable latency (email, SMS, webhook
  callbacks), poll the actual source system (e.g. a test mailbox API)
  instead of waiting on a UI element with a fixed timeout.
