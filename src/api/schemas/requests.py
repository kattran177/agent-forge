"""
Request body schemas for AgentForge API endpoints.

Each Pydantic model defines:
- Required vs optional fields
- Type constraints (str, int, list, etc.)
- Field-level validation (min/max length, regex, allowed values)
- OpenAPI documentation via `description` and `examples`

FastAPI uses these models to:
1. Validate incoming requests automatically
2. Return a 422 Unprocessable Entity on bad input (with a clear error body)
3. Generate accurate OpenAPI/Swagger docs
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ── Allowed values ─────────────────────────────────────────────────

ALLOWED_AGENT_MODES = Literal["research", "write", "plan", "full"]


# ── Request models ─────────────────────────────────────────────────


class GoalRequest(BaseModel):
    """
    Legacy single-field request for backwards compatibility.
    Prefer AgentRunRequest for new integrations.
    """

    goal: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="The high-level goal for the agent system to accomplish",
        examples=["Research the top 3 AI agent frameworks and compare them"],
    )


class AgentRunRequest(BaseModel):
    """
    Full request body for POST /api/v1/tasks.

    Controls which agents run and how they behave.
    """

    goal: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="The high-level goal for the agent system to accomplish",
        examples=["Research the top 3 AI agent frameworks and compare them"],
    )
    mode: ALLOWED_AGENT_MODES = Field(
        default="full",
        description=(
            "Which agents to activate: "
            "'research' (researcher only), "
            "'write' (writer only), "
            "'plan' (planner only), "
            "'full' (all agents — default)"
        ),
    )
    max_steps: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of agent steps before halting (1–50)",
    )
    conversation_id: str | None = Field(
        default=None,
        max_length=64,
        description="Optional ID to continue an existing conversation",
        examples=["conv_abc123"],
    )

    @field_validator("goal")
    @classmethod
    def goal_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("goal must not be blank or whitespace only")
        return v.strip()


class ConversationRequest(BaseModel):
    """
    Request body for POST /api/v1/conversations/{conversation_id}/messages.

    Sends a follow-up message inside an ongoing conversation.
    """

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's follow-up message",
        examples=["Can you expand on the LangGraph section?"],
    )
    role: Literal["user", "assistant", "system"] = Field(
        default="user",
        description="The role of the message sender",
    )

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be blank or whitespace only")
        return v.strip()
