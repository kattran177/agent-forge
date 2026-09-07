"""
Tests for Pydantic request validation on AgentForge API endpoints.

These tests verify that:
- Valid input is accepted and parsed correctly
- Missing required fields return 422 Unprocessable Entity
- Fields that exceed max_length constraints return 422
- Fields with invalid enum values return 422
- Whitespace-only fields are rejected by custom validators
- Optional fields default to the correct values

All tests use FastAPI's ASGI test transport via httpx — no real
HTTP server is needed, so the suite runs fast and offline.
"""

from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import app
from src.api.schemas.requests import AgentRunRequest, ConversationRequest


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def client():
    """Async HTTP client wired directly to the FastAPI app."""
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


# ── Unit tests: model-level validation (no HTTP overhead) ─────────


class TestAgentRunRequestModel:
    """Direct model validation — fast, no HTTP round-trip."""

    def test_valid_request_defaults(self):
        """A goal with only required fields should use sensible defaults."""
        req = AgentRunRequest(goal="Research the latest AI agent frameworks")
        assert req.mode == "full"
        assert req.max_steps == 10
        assert req.conversation_id is None

    def test_goal_too_short_raises(self):
        """Goals under 10 characters must be rejected."""
        with pytest.raises(Exception):
            AgentRunRequest(goal="short")

    def test_goal_too_long_raises(self):
        """Goals over 1000 characters must be rejected."""
        with pytest.raises(Exception):
            AgentRunRequest(goal="x" * 1001)

    def test_goal_whitespace_only_raises(self):
        """A goal containing only whitespace must be rejected by the custom validator."""
        with pytest.raises(Exception):
            AgentRunRequest(goal="          ")  # 10 spaces — passes length but not blank check

    def test_goal_is_stripped(self):
        """Leading/trailing whitespace should be stripped from the goal."""
        req = AgentRunRequest(goal="  Research AI agent frameworks today  ")
        assert req.goal == "Research AI agent frameworks today"

    def test_invalid_mode_raises(self):
        """An unrecognised mode value must be rejected."""
        with pytest.raises(Exception):
            AgentRunRequest(goal="Research the latest AI agent frameworks", mode="invalid")

    def test_valid_modes_accepted(self):
        """All documented mode values must be accepted."""
        for mode in ("research", "write", "plan", "full"):
            req = AgentRunRequest(goal="Research the latest AI agent frameworks", mode=mode)
            assert req.mode == mode

    def test_max_steps_bounds(self):
        """max_steps must be between 1 and 50 inclusive."""
        with pytest.raises(Exception):
            AgentRunRequest(goal="Research the latest AI agent frameworks", max_steps=0)
        with pytest.raises(Exception):
            AgentRunRequest(goal="Research the latest AI agent frameworks", max_steps=51)
        # Boundary values should succeed
        req_low = AgentRunRequest(goal="Research the latest AI agent frameworks", max_steps=1)
        req_high = AgentRunRequest(goal="Research the latest AI agent frameworks", max_steps=50)
        assert req_low.max_steps == 1
        assert req_high.max_steps == 50

    def test_conversation_id_too_long_raises(self):
        """conversation_id longer than 64 characters must be rejected."""
        with pytest.raises(Exception):
            AgentRunRequest(
                goal="Research the latest AI agent frameworks",
                conversation_id="c" * 65,
            )


class TestConversationRequestModel:
    """Direct validation tests for ConversationRequest."""

    def test_valid_message(self):
        req = ConversationRequest(message="Can you expand on the LangGraph section?")
        assert req.role == "user"

    def test_empty_message_raises(self):
        with pytest.raises(Exception):
            ConversationRequest(message="")

    def test_whitespace_only_message_raises(self):
        with pytest.raises(Exception):
            ConversationRequest(message="   ")

    def test_message_too_long_raises(self):
        with pytest.raises(Exception):
            ConversationRequest(message="x" * 2001)

    def test_invalid_role_raises(self):
        with pytest.raises(Exception):
            ConversationRequest(message="Hello", role="robot")

    def test_valid_roles_accepted(self):
        for role in ("user", "assistant", "system"):
            req = ConversationRequest(message="Hello there", role=role)
            assert req.role == role


# ── Integration tests: HTTP endpoint validation (422 responses) ───


class TestTaskEndpointValidation:
    """HTTP-level validation — verifies FastAPI returns correct status codes."""

    @pytest.mark.asyncio
    async def test_missing_goal_returns_422(self, client):
        """Omitting the required `goal` field must return 422."""
        response = await client.post("/api/v1/tasks", json={})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_goal_too_short_returns_422(self, client):
        """A goal under 10 characters must return 422."""
        response = await client.post("/api/v1/tasks", json={"goal": "short"})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_goal_too_long_returns_422(self, client):
        """A goal over 1000 characters must return 422."""
        response = await client.post("/api/v1/tasks", json={"goal": "x" * 1001})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_mode_returns_422(self, client):
        """An invalid mode value must return 422."""
        response = await client.post(
            "/api/v1/tasks",
            json={"goal": "Research the latest AI agent frameworks", "mode": "invalid"},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_max_steps_out_of_range_returns_422(self, client):
        """max_steps=0 (below minimum) must return 422."""
        response = await client.post(
            "/api/v1/tasks",
            json={"goal": "Research the latest AI agent frameworks", "max_steps": 0},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    @patch("src.api.main.run_agent")
    async def test_valid_request_returns_200(self, mock_run_agent, client):
        """A fully valid request must be accepted and return 200."""
        mock_run_agent.return_value = {
            "status": "complete",
            "plan": {},
            "research_findings": "findings",
            "final_report": "report",
            "activity_log": [],
            "error": "",
        }
        response = await client.post(
            "/api/v1/tasks",
            json={
                "goal": "Research the top 3 AI agent frameworks and compare them",
                "mode": "full",
                "max_steps": 5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "complete"

    @pytest.mark.asyncio
    @patch("src.api.main.run_agent")
    async def test_422_error_body_contains_detail(self, mock_run_agent, client):
        """422 responses must include a `detail` array describing the error."""
        response = await client.post("/api/v1/tasks", json={"goal": "hi"})
        assert response.status_code == 422
        body = response.json()
        assert "detail" in body
        # detail should be a list of validation errors
        assert isinstance(body["detail"], list)
        assert len(body["detail"]) > 0
