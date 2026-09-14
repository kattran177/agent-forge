"""
Request context — stores per-request state using Python's contextvars.

WHY contextvars?
Python's threading.local() works for synchronous code, but async frameworks
like FastAPI use coroutines instead of threads. A contextvars.ContextVar
is scoped to the current async task (coroutine), just as threading.local()
is scoped to a thread. This means each concurrent request gets its own
isolated copy of the variable — no cross-request contamination.

USAGE:
    # In middleware (set once per request):
    token = correlation_id_var.set("req-abc123")

    # Anywhere downstream (agents, tools, etc.):
    cid = get_correlation_id()   # → "req-abc123"

    # The ContextVar automatically propagates into sub-coroutines spawned
    # from the same async context, so all log lines from one request share
    # the same correlation ID with zero extra plumbing.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar, Token

# The sentinel value used when no correlation ID has been set (e.g. in tests
# or background tasks that were not spawned from an HTTP request).
_UNSET = "no-correlation-id"

correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default=_UNSET)


def set_correlation_id(value: str | None = None) -> Token:
    """
    Set the correlation ID for the current context.

    If *value* is not provided, a fresh UUID4 short ID is generated.
    Returns the ``Token`` from ``ContextVar.set()`` so callers can restore
    the previous value when needed (e.g. in tests).
    """
    if value is None:
        value = _generate_id()
    return correlation_id_var.set(value)


def get_correlation_id() -> str:
    """Return the correlation ID for the current context."""
    return correlation_id_var.get()


def _generate_id() -> str:
    """Generate a short, URL-safe correlation ID (first 12 hex chars of UUID4)."""
    return uuid.uuid4().hex[:12]
