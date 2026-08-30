"""
Tests for the MemoryStore.
"""

import os
import tempfile

import pytest

from src.memory.store import MemoryStore, TaskRecord


@pytest.fixture
def memory():
    """Create a temporary memory store for testing."""
    # Use a temp file so tests don't pollute each other
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store = MemoryStore(db_path=path)
    yield store
    os.unlink(path)  # Cleanup after test


class TestMemoryStore:
    """Test the SQLite memory store."""

    def test_save_and_get_task(self, memory):
        """Should be able to save and retrieve a task."""
        task = TaskRecord(
            task_id="test-1",
            goal="Research AI",
            agent="researcher",
            input_text="Find AI papers",
            output_text="Found 5 relevant papers",
            status="completed",
        )
        memory.save_task(task)

        retrieved = memory.get_task("test-1")
        assert retrieved is not None
        assert retrieved.goal == "Research AI"
        assert retrieved.status == "completed"
        assert retrieved.output_text == "Found 5 relevant papers"

    def test_get_nonexistent_task(self, memory):
        """Getting a non-existent task should return None."""
        assert memory.get_task("nonexistent") is None

    def test_update_task(self, memory):
        """Saving a task with the same ID should update it."""
        task = TaskRecord(
            task_id="test-2",
            goal="Test",
            agent="planner",
            input_text="Plan",
            output_text="",
            status="pending",
        )
        memory.save_task(task)

        task.status = "completed"
        task.output_text = "Done"
        memory.save_task(task)

        retrieved = memory.get_task("test-2")
        assert retrieved.status == "completed"
        assert retrieved.output_text == "Done"

    def test_get_tasks_for_goal(self, memory):
        """Should return all tasks for a specific goal."""
        for i in range(3):
            memory.save_task(
                TaskRecord(
                    task_id=f"goal-task-{i}",
                    goal="Big Goal",
                    agent="researcher",
                    input_text=f"Task {i}",
                    output_text=f"Result {i}",
                    status="completed",
                )
            )

        tasks = memory.get_tasks_for_goal("Big Goal")
        assert len(tasks) == 3

    def test_save_and_get_conversation(self, memory):
        """Should store and retrieve conversation messages."""
        memory.save_message("session-1", "user", "Hello")
        memory.save_message("session-1", "assistant", "Hi there!")

        messages = memory.get_conversation("session-1")
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_conversation_isolation(self, memory):
        """Messages from different sessions should not mix."""
        memory.save_message("session-a", "user", "Message A")
        memory.save_message("session-b", "user", "Message B")

        messages_a = memory.get_conversation("session-a")
        assert len(messages_a) == 1
        assert messages_a[0]["content"] == "Message A"
