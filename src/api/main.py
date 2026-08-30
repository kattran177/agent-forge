"""
FastAPI REST API for AgentForge.

WHY AN API?
A web API makes your agent system accessible to:
- Frontend applications (React, Streamlit, mobile apps)
- Other services (microservice architecture)
- Command-line tools (curl, httpie)
- Automated workflows (CI/CD, cron jobs)

FastAPI is the modern standard for Python APIs because:
- Async by default (handles concurrent requests)
- Auto-generates OpenAPI docs (try /docs in browser)
- Type validation with Pydantic (catches bad input)
- Very fast (built on Starlette + Uvicorn)
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agents.orchestrator import run_agent
from src.memory.store import MemoryStore


# ── Models (request/response schemas) ─────────────────────────────
# Pydantic models define the shape of API requests and responses.
# FastAPI uses these for validation AND auto-generated documentation.


class GoalRequest(BaseModel):
    """Request body for submitting a goal to the agent system."""

    goal: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="The high-level goal for the agent system to accomplish",
        examples=["Research the top 3 AI agent frameworks and compare them"],
    )


class TaskResponse(BaseModel):
    """Response body with the agent execution results."""

    goal: str
    status: str
    plan: dict
    research_findings: str
    final_report: str
    activity_log: list
    error: str = ""


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str


# ── Application setup ──────────────────────────────────────────────

memory = MemoryStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    print("[INFO] AgentForge API starting...")
    yield
    print("[INFO] AgentForge API shutting down...")


app = FastAPI(
    title="AgentForge API",
    description="Multi-agent AI task orchestration platform",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Endpoints ──────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint — used by Docker/Kubernetes to verify the service is alive."""
    return HealthResponse(status="healthy", version="1.0.0")


@app.post("/api/v1/tasks", response_model=TaskResponse)
async def create_task(request: GoalRequest):
    """
    Submit a goal for the agent system to accomplish.

    The system will:
    1. Plan: Break the goal into subtasks
    2. Research: Gather information from the web
    3. Write: Synthesise findings into a report

    This is a synchronous endpoint — it blocks until the agents finish.
    For production, you'd use a task queue (Celery/Redis) and return
    a task ID that the client can poll for results.
    """
    try:
        result = run_agent(request.goal, memory=memory)

        return TaskResponse(
            goal=request.goal,
            status=result.get("status", "unknown"),
            plan=result.get("plan", {}),
            research_findings=result.get("research_findings", ""),
            final_report=result.get("final_report", ""),
            activity_log=result.get("activity_log", []),
            error=result.get("error", ""),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/tasks", response_model=list[dict])
async def list_tasks():
    """List recent tasks from memory."""
    # Get all tasks (simplified — in production you'd add pagination)
    with __import__("sqlite3").connect(memory.db_path) as conn:
        conn.row_factory = __import__("sqlite3").Row
        rows = conn.execute(
            "SELECT task_id, goal, status, created_at FROM tasks ORDER BY created_at DESC LIMIT 20"
        ).fetchall()

    return [dict(r) for r in rows]
