"""Response body schemas for AgentForge API endpoints."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str


class TaskResponse(BaseModel):
    """Response body with agent execution results."""

    goal: str
    status: str
    plan: dict
    research_findings: str
    final_report: str
    activity_log: list
    error: str = ""


class MessageResponse(BaseModel):
    """Response for a single conversation message."""

    conversation_id: str = Field(..., description="ID of the conversation")
    role: str = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Text content of the message")
