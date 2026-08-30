"""
SQLite-based conversation and task memory.

WHY SQLITE FOR MEMORY?
Agents need to remember:
- What goal they're working on
- What tasks have been completed
- What information has been gathered
- Previous conversation turns

SQLite is perfect here because:
- Zero setup (it's a file, no server needed)
- Included with Python (no extra dependencies)
- Fast enough for agent memory (not high-concurrency)
- Easy to inspect and debug (just open the .db file)

In production, you'd swap this for PostgreSQL or Redis for multi-user support.
"""

import json
import sqlite3
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional

from src.config import SQLITE_PATH


@dataclass
class TaskRecord:
    """A record of a task and its result in agent memory."""

    task_id: str
    goal: str
    agent: str  # Which agent handled this task
    input_text: str
    output_text: str
    status: str  # "pending", "running", "completed", "failed"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict = field(default_factory=dict)


class MemoryStore:
    """
    Persistent memory for agent conversations and task results.

    Uses SQLite as a simple key-value + relational store.
    All agent outputs are stored here so:
    1. Agents can reference each other's work
    2. Users can see the full execution history
    3. The system can resume after a crash
    """

    def __init__(self, db_path: str = SQLITE_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    goal TEXT NOT NULL,
                    agent TEXT NOT NULL,
                    input_text TEXT NOT NULL,
                    output_text TEXT DEFAULT '',
                    status TEXT DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
            conn.commit()

    def save_task(self, task: TaskRecord) -> None:
        """Save or update a task record."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO tasks
                    (task_id, goal, agent, input_text, output_text, status, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.task_id,
                    task.goal,
                    task.agent,
                    task.input_text,
                    task.output_text,
                    task.status,
                    task.created_at,
                    json.dumps(task.metadata),
                ),
            )
            conn.commit()

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        """Retrieve a task by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()

        if not row:
            return None

        return TaskRecord(
            task_id=row["task_id"],
            goal=row["goal"],
            agent=row["agent"],
            input_text=row["input_text"],
            output_text=row["output_text"],
            status=row["status"],
            created_at=row["created_at"],
            metadata=json.loads(row["metadata"]),
        )

    def get_tasks_for_goal(self, goal: str) -> list[TaskRecord]:
        """Get all tasks associated with a goal."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM tasks WHERE goal = ? ORDER BY created_at", (goal,)
            ).fetchall()

        return [
            TaskRecord(
                task_id=r["task_id"],
                goal=r["goal"],
                agent=r["agent"],
                input_text=r["input_text"],
                output_text=r["output_text"],
                status=r["status"],
                created_at=r["created_at"],
                metadata=json.loads(r["metadata"]),
            )
            for r in rows
        ]

    def save_message(self, session_id: str, role: str, content: str) -> None:
        """Save a conversation message."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO conversations (session_id, role, content, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, content, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()

    def get_conversation(self, session_id: str, limit: int = 50) -> list[dict]:
        """Get recent conversation messages for a session."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT role, content, timestamp FROM conversations
                WHERE session_id = ?
                ORDER BY id DESC LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()

        # Return in chronological order
        return [
            {"role": r["role"], "content": r["content"], "timestamp": r["timestamp"]}
            for r in reversed(rows)
        ]
