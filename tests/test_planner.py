"""
Tests for the Planner agent.
"""

import json
from unittest.mock import patch

from src.agents.planner import create_plan


class TestPlanner:
    """Test goal decomposition into subtasks."""

    @patch("src.agents.planner.chat")
    def test_create_plan_valid_json(self, mock_chat):
        """Planner should return a structured plan with subtasks."""
        mock_chat.return_value = json.dumps(
            {
                "goal": "Test goal",
                "subtasks": [
                    {"id": "1", "description": "Research topic", "agent": "researcher"},
                    {"id": "2", "description": "Write report", "agent": "writer"},
                ],
            }
        )

        plan = create_plan("Test goal")

        assert plan["goal"] == "Test goal"
        assert len(plan["subtasks"]) == 2
        assert plan["subtasks"][0]["agent"] == "researcher"
        assert plan["subtasks"][1]["agent"] == "writer"

    @patch("src.agents.planner.chat")
    def test_create_plan_with_code_block(self, mock_chat):
        """Planner should handle JSON wrapped in markdown code blocks."""
        mock_chat.return_value = '```json\n{"goal": "Test", "subtasks": [{"id": "1", "description": "Do thing", "agent": "researcher"}]}\n```'

        plan = create_plan("Test")

        assert "subtasks" in plan
        assert len(plan["subtasks"]) == 1

    @patch("src.agents.planner.chat")
    def test_create_plan_invalid_json_fallback(self, mock_chat):
        """Planner should return a fallback plan if JSON parsing fails."""
        mock_chat.return_value = "This is not valid JSON at all"

        plan = create_plan("Research AI")

        # Should fall back to a default 2-step plan
        assert plan["goal"] == "Research AI"
        assert len(plan["subtasks"]) >= 1

    @patch("src.agents.planner.chat")
    def test_create_plan_missing_fields(self, mock_chat):
        """Planner should handle missing fields gracefully."""
        mock_chat.return_value = json.dumps({"subtasks": [{"id": "1", "description": "Do research", "agent": "researcher"}]})

        plan = create_plan("My goal")

        assert plan["goal"] == "My goal"  # Should be filled in
        assert "subtasks" in plan


class TestPlannerPrompt:
    """Test that the planner prompt produces the right structure."""

    @patch("src.agents.planner.chat")
    def test_planner_calls_llm_with_system_prompt(self, mock_chat):
        """Planner should send a system prompt and user message."""
        mock_chat.return_value = json.dumps(
            {"goal": "x", "subtasks": []}
        )

        create_plan("Research AI agents")

        # Verify chat was called with messages
        mock_chat.assert_called_once()
        messages = mock_chat.call_args[0][0]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Research AI agents" in messages[1]["content"]
