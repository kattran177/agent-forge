"""
Structured JSON logging for AgentForge.

WHY STRUCTURED LOGGING?
Plain text logs are fine for humans reading a terminal.  But in production:
- You ship logs to a centralised system (Datadog, CloudWatch, Loki)
- You need to search/filter by field: level=ERROR, agent=researcher
- You need to correlate a *single request* across many log lines

JSON logs make all of that trivial — every log line is a parseable record.

OUTPUT FORMAT (one JSON object per line):
    {
      "timestamp": "2026-09-14T18:42:00.123456Z",
      "level": "INFO",
      "correlation_id": "a3f9b1c2d4e5",
      "agent_name": "planner",
      "message": "Plan created with 3 subtasks"
    }

USAGE:
    from src.core.logging import get_logger

    logger = get_logger("planner")
    logger.info("Plan created", subtask_count=3)
    logger.warning("LLM returned unexpected format")
    logger.error("Research failed", error=str(exc))

Extra keyword arguments are merged into the JSON record as additional fields,
which is useful for structured context (task IDs, durations, counts, etc.).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from src.core.context import get_correlation_id


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------

class _JSONFormatter(logging.Formatter):
    """
    Formats a LogRecord as a single-line JSON object.

    Standard fields emitted on every line:
      timestamp      ISO-8601 UTC
      level          CRITICAL / ERROR / WARNING / INFO / DEBUG
      correlation_id pulled from the current ContextVar
      agent_name     set via the "agent_name" extra field
      message        the formatted log message

    Any extra fields passed via logging.Logger.info(msg, extra={...}) or
    via the LoggerAdapter in get_logger() are merged into the record.
    """

    # Fields that exist on every LogRecord but should not bleed into the
    # JSON output as duplicates.
    _SKIP = frozenset({
        "args", "created", "exc_info", "exc_text", "filename", "funcName",
        "levelname", "levelno", "lineno", "message", "module", "msecs",
        "msg", "name", "pathname", "process", "processName", "relativeCreated",
        "stack_info", "taskName", "thread", "threadName",
    })

    def format(self, record: logging.LogRecord) -> str:
        # Build the base record first (fills record.message)
        record.message = record.getMessage()

        doc: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.%f"
            )[:-3] + "Z",
            "level": record.levelname,
            "correlation_id": get_correlation_id(),
            "agent_name": getattr(record, "agent_name", record.name),
            "message": record.message,
        }

        # Merge any extra fields the caller attached
        for key, value in record.__dict__.items():
            if key not in self._SKIP and not key.startswith("_"):
                doc[key] = value

        # Append exception traceback as a string field if present
        if record.exc_info:
            doc["exception"] = self.formatException(record.exc_info)

        return json.dumps(doc, default=str)


# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

def configure_logging(level: str = "INFO") -> None:
    """
    Configure the root logger to emit structured JSON to stdout.

    Call this once at application startup (in main.py lifespan, CLI entry point,
    etc.).  Calling it multiple times is safe — handlers are not duplicated.
    """
    root = logging.getLogger()

    # Avoid adding duplicate handlers if configure_logging is called again
    if any(isinstance(h, logging.StreamHandler) and isinstance(h.formatter, _JSONFormatter)
           for h in root.handlers):
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JSONFormatter())
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Silence noisy third-party loggers that spam at DEBUG level
    for noisy in ("httpx", "httpcore", "groq", "urllib3", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Per-agent logger factory
# ---------------------------------------------------------------------------

class _AgentLoggerAdapter(logging.LoggerAdapter):
    """
    A LoggerAdapter that injects ``agent_name`` into every log record and
    supports keyword arguments as extra structured fields.

    Usage::

        logger = get_logger("planner")
        logger.info("Plan created", subtask_count=3, goal="...")
    """

    def __init__(self, logger: logging.Logger, agent_name: str) -> None:
        super().__init__(logger, extra={"agent_name": agent_name})
        self._agent_name = agent_name

    # Override process() so callers can pass arbitrary keyword args as extras
    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:
        extra = dict(self.extra)  # start with agent_name

        # Pull keyword args out of the call and stash them as extra fields
        call_extras = kwargs.pop("extra", {})
        # Also support passing fields as plain keyword args to info()/error()/etc.
        # We must not pass unknown kwargs to the underlying logger, so we
        # intercept them here.
        for key in list(kwargs):
            if key not in ("exc_info", "stack_info", "stacklevel"):
                extra[key] = kwargs.pop(key)

        extra.update(call_extras)
        kwargs["extra"] = extra
        return msg, kwargs

    # Convenience: support logger.info("msg", key=value) call style
    def _log_with_extras(self, level: int, msg: str, *args: Any, **kwargs: Any) -> None:
        if self.isEnabledFor(level):
            self._log(level, msg, args, **kwargs)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        self._log_with_extras(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        self._log_with_extras(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        self._log_with_extras(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        self._log_with_extras(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        self._log_with_extras(logging.CRITICAL, msg, *args, **kwargs)

    def _log(self, level: int, msg: str, args: tuple, **kwargs: Any) -> None:
        """Bridge to the underlying logger's _log method after processing kwargs."""
        msg, kwargs = self.process(msg, kwargs)
        self.logger._log(level, msg, args, **kwargs)


def get_logger(agent_name: str) -> _AgentLoggerAdapter:
    """
    Return a structured logger bound to the given agent name.

    The returned logger automatically includes:
    - ``agent_name`` in every JSON record
    - ``correlation_id`` pulled live from the current context

    Args:
        agent_name: Short name for the component, e.g. ``"planner"``,
                    ``"researcher"``, ``"api"``.
    """
    base = logging.getLogger(f"agentforge.{agent_name}")
    return _AgentLoggerAdapter(base, agent_name)
