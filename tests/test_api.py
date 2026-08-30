"""
Tests for the FastAPI endpoints.
"""

from unittest.mock import patch

import pytest
from httpx import AsyncClient, ASGITransport

from src.api.main import app


@pytest.fixture
def client():
    """Create a test client."""
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


class TestHealthEndpoint:
    """Test the health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_check(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestTaskEndpoint:
    """Test the task creation endpoint."""

    @pytest.mark.asyncio
    async def test_create_task_validation(self, client):
        """Short goals should be rejected."""
        response = await client.post(
            "/api/v1/tasks",
            json={"goal": "short"},  # Less than 10 chars
        )
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    @patch("src.api.main.run_agent")
    async def test_create_task_success(self, mock_run, client):
        """Valid goal should trigger agent execution."""
        mock_run.return_value = {
            "goal": "Research AI agent frameworks",
            "status": "complete",
            "plan": {"subtasks": []},
            "research_findings": "Some findings",
            "final_report": "A great report",
            "activity_log": [],
            "error": "",
        }

        response = await client.post(
            "/api/v1/tasks",
            json={"goal": "Research the top AI agent frameworks and compare them"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "complete"
        assert data["final_report"] == "A great report"
