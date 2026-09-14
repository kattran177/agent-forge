"""
Core utilities for AgentForge.

Exports:
    get_logger         — structured JSON logger bound to an agent name
    configure_logging  — one-time logging setup (call at startup)
    get_correlation_id — read the current request's correlation ID
    set_correlation_id — write a correlation ID into the current context
"""

from src.core.context import get_correlation_id, set_correlation_id
from src.core.logging import configure_logging, get_logger

__all__ = [
    "get_logger",
    "configure_logging",
    "get_correlation_id",
    "set_correlation_id",
]
