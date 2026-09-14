"""
Correlation ID middleware for FastAPI.

WHAT THIS DOES:
Every HTTP request that arrives at the API gets a unique correlation ID.
The ID is:
  1. Read from the incoming ``X-Correlation-ID`` header (if the caller supplies one)
     This lets upstream services propagate their own trace IDs, which is
     important in microservice architectures where a single user action
     triggers calls to multiple services.
  2. Generated fresh (UUID4 short form) if no header is present.

The ID is then:
  - Stored in the ``correlation_id_var`` ContextVar so every log line
    emitted during request processing automatically includes it.
  - Written to the ``X-Correlation-ID`` response header so callers can
    correlate their own logs against ours.

WHY STARLETTE BaseHTTPMiddleware?
FastAPI is built on Starlette. ``BaseHTTPMiddleware`` is the standard way
to wrap every request/response cycle.  It runs *before* any route handler
and *after* the response is built, giving us a clean place to set up and
tear down per-request state.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.core.context import set_correlation_id, get_correlation_id
from src.core.logging import get_logger

_logger = get_logger("middleware")

CORRELATION_ID_HEADER = "X-Correlation-ID"


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware that attaches a correlation ID to every request.

    The ID is taken from the ``X-Correlation-ID`` request header when
    present, or generated as a fresh UUID4 short form otherwise.
    It is stored in a ``ContextVar`` for the duration of the request and
    echoed back in the response headers.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # Accept caller-supplied ID or generate a fresh one
        incoming = request.headers.get(CORRELATION_ID_HEADER)
        token = set_correlation_id(incoming)  # sets ContextVar for this coroutine
        correlation_id = get_correlation_id()

        _logger.info(
            "Request started",
            method=request.method,
            path=request.url.path,
        )

        try:
            response: Response = await call_next(request)
        except Exception as exc:
            _logger.error(
                "Unhandled exception during request",
                method=request.method,
                path=request.url.path,
                error=str(exc),
            )
            raise
        finally:
            # Reset the ContextVar to its previous value.  This is good practice
            # even though each request runs in its own coroutine scope, because
            # tests re-use the same event loop and context.
            from src.core.context import correlation_id_var
            correlation_id_var.reset(token)

        _logger.info(
            "Request completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
        )

        # Propagate the ID back to the caller
        response.headers[CORRELATION_ID_HEADER] = correlation_id
        return response
