"""
Tests for the structured logging and correlation ID middleware.

Coverage:
  - context.py: set/get correlation ID, isolation between contexts
  - logging.py: JSON output format, required fields, extra fields, configure_logging idempotency
  - middleware.py: ID generated when header absent, header propagated when present,
                   ID echoed in response header, existing tests still pass
"""

from __future__ import annotations

import json
import logging
import io
from unittest.mock import patch

import pytest
from httpx import AsyncClient, ASGITransport

from src.core.context import (
    correlation_id_var,
    get_correlation_id,
    set_correlation_id,
    _UNSET,
)
from src.core.logging import _JSONFormatter, configure_logging, get_logger
from src.api.middleware import CORRELATION_ID_HEADER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _capture_log_record(level: str, message: str, **extra) -> dict:
    """
    Emit one log record through _JSONFormatter and return the parsed JSON dict.
    """
    formatter = _JSONFormatter()
    record = logging.LogRecord(
        name="agentforge.test",
        level=getattr(logging, level.upper()),
        pathname=__file__,
        lineno=0,
        msg=message,
        args=(),
        exc_info=None,
    )
    for k, v in extra.items():
        setattr(record, k, v)
    output = formatter.format(record)
    return json.loads(output)


# ---------------------------------------------------------------------------
# context.py tests
# ---------------------------------------------------------------------------

class TestCorrelationIDContext:

    def test_default_value_is_unset_sentinel(self):
        # Reset to default before testing
        token = correlation_id_var.set(_UNSET)
        try:
            assert get_correlation_id() == _UNSET
        finally:
            correlation_id_var.reset(token)

    def test_set_generates_id_when_none_given(self):
        token = set_correlation_id()
        try:
            cid = get_correlation_id()
            assert cid != _UNSET
            assert len(cid) == 12  # UUID4 hex[:12]
            assert cid.isalnum()
        finally:
            correlation_id_var.reset(token)

    def test_set_uses_provided_value(self):
        token = set_correlation_id("my-custom-id")
        try:
            assert get_correlation_id() == "my-custom-id"
        finally:
            correlation_id_var.reset(token)

    def test_reset_restores_previous_value(self):
        outer_token = set_correlation_id("outer-id")
        inner_token = set_correlation_id("inner-id")
        assert get_correlation_id() == "inner-id"

        correlation_id_var.reset(inner_token)
        assert get_correlation_id() == "outer-id"

        correlation_id_var.reset(outer_token)

    def test_two_independent_sets_are_distinct(self):
        t1 = set_correlation_id()
        id1 = get_correlation_id()
        correlation_id_var.reset(t1)

        t2 = set_correlation_id()
        id2 = get_correlation_id()
        correlation_id_var.reset(t2)

        assert id1 != id2  # Extremely unlikely to collide with UUID4


# ---------------------------------------------------------------------------
# logging.py — _JSONFormatter tests
# ---------------------------------------------------------------------------

class TestJSONFormatter:

    def test_output_is_valid_json(self):
        record = _capture_log_record("INFO", "hello world")
        assert isinstance(record, dict)

    def test_required_fields_present(self):
        doc = _capture_log_record("INFO", "test message")
        for field in ("timestamp", "level", "correlation_id", "agent_name", "message"):
            assert field in doc, f"Missing field: {field}"

    def test_level_is_uppercased(self):
        doc = _capture_log_record("warning", "watch out")
        assert doc["level"] == "WARNING"

    def test_message_matches(self):
        doc = _capture_log_record("INFO", "exact message text")
        assert doc["message"] == "exact message text"

    def test_agent_name_from_extra(self):
        doc = _capture_log_record("INFO", "msg", agent_name="planner")
        assert doc["agent_name"] == "planner"

    def test_correlation_id_from_context(self):
        token = set_correlation_id("test-cid-abc")
        try:
            doc = _capture_log_record("INFO", "msg")
            assert doc["correlation_id"] == "test-cid-abc"
        finally:
            correlation_id_var.reset(token)

    def test_extra_fields_included(self):
        doc = _capture_log_record("INFO", "msg", subtask_count=3, goal="test goal")
        assert doc["subtask_count"] == 3
        assert doc["goal"] == "test goal"

    def test_timestamp_format(self):
        doc = _capture_log_record("INFO", "msg")
        ts = doc["timestamp"]
        # Must be ISO-8601 with milliseconds and Z suffix: 2026-09-14T18:42:00.123Z
        assert ts.endswith("Z")
        assert "T" in ts
        assert len(ts) == 24  # "YYYY-MM-DDTHH:MM:SS.mmmZ"

    def test_exception_info_included(self):
        formatter = _JSONFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            exc = sys.exc_info()
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname=__file__,
            lineno=0, msg="err", args=(), exc_info=exc,
        )
        output = json.loads(formatter.format(record))
        assert "exception" in output
        assert "ValueError" in output["exception"]


# ---------------------------------------------------------------------------
# logging.py — configure_logging idempotency
# ---------------------------------------------------------------------------

class TestConfigureLogging:

    def test_calling_twice_does_not_duplicate_handlers(self):
        root = logging.getLogger()
        before = len(root.handlers)
        configure_logging()
        configure_logging()  # second call — must be idempotent
        after = len(root.handlers)
        # Handler count should not grow on the second call
        assert after == before or after == before + 1  # first call may add one


# ---------------------------------------------------------------------------
# logging.py — get_logger / _AgentLoggerAdapter
# ---------------------------------------------------------------------------

class TestGetLogger:

    def test_returns_adapter(self):
        from src.core.logging import _AgentLoggerAdapter
        logger = get_logger("test-agent")
        assert isinstance(logger, _AgentLoggerAdapter)

    def test_info_emits_json_with_agent_name(self):
        configure_logging()
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(_JSONFormatter())

        logger = get_logger("test-agent-x")
        logger.logger.addHandler(handler)
        logger.logger.setLevel(logging.DEBUG)

        logger.info("hello from test")
        output = stream.getvalue().strip()

        doc = json.loads(output)
        assert doc["agent_name"] == "test-agent-x"
        assert doc["message"] == "hello from test"

        logger.logger.removeHandler(handler)

    def test_extra_kwargs_appear_in_output(self):
        configure_logging()
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(_JSONFormatter())

        logger = get_logger("test-extra")
        logger.logger.addHandler(handler)
        logger.logger.setLevel(logging.DEBUG)

        logger.info("structured", task="my-task", count=7)
        output = stream.getvalue().strip()
        doc = json.loads(output)
        assert doc.get("task") == "my-task"
        assert doc.get("count") == 7

        logger.logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# middleware.py — HTTP-level tests
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    from src.api.main import app
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


class TestCorrelationIDMiddleware:

    @pytest.mark.anyio
    async def test_response_contains_correlation_id_header(self, client):
        response = await client.get("/health")
        assert CORRELATION_ID_HEADER in response.headers

    @pytest.mark.anyio
    async def test_generated_id_is_12_hex_chars(self, client):
        response = await client.get("/health")
        cid = response.headers[CORRELATION_ID_HEADER]
        assert len(cid) == 12
        assert cid.isalnum()

    @pytest.mark.anyio
    async def test_supplied_id_is_echoed_back(self, client):
        custom_id = "my-trace-id-99"
        response = await client.get(
            "/health",
            headers={CORRELATION_ID_HEADER: custom_id},
        )
        assert response.headers[CORRELATION_ID_HEADER] == custom_id

    @pytest.mark.anyio
    async def test_each_request_gets_unique_id(self, client):
        r1 = await client.get("/health")
        r2 = await client.get("/health")
        assert r1.headers[CORRELATION_ID_HEADER] != r2.headers[CORRELATION_ID_HEADER]

    @pytest.mark.anyio
    async def test_health_endpoint_still_works(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    @pytest.mark.anyio
    @patch("src.api.main.run_agent")
    async def test_task_endpoint_still_works(self, mock_run, client):
        mock_run.return_value = {
            "status": "complete",
            "plan": {"subtasks": []},
            "research_findings": "",
            "final_report": "Done",
            "activity_log": [],
            "error": "",
        }
        response = await client.post(
            "/api/v1/tasks",
            json={"goal": "Research the top AI agent frameworks and compare them"},
        )
        assert response.status_code == 200
        assert CORRELATION_ID_HEADER in response.headers
