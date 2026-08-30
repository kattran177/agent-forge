"""
Planner Agent — decomposes high-level goals into ordered subtasks.

HOW PLANNING WORKS IN AGENTIC AI:
The key insight behind AI agents is that LLMs are surprisingly good at
breaking down complex tasks into steps — IF you give them the right prompt.

This agent takes a goal like:
  "Research the top AI agent frameworks and compare them"

And produces a structured plan:
  1. Search for "top AI agent frameworks 2026"
  2. Gather details on the top 3 frameworks
  3. Compare their features, pros, and cons
  4. Write a structured comparison report

The Orchestrator then executes each step using the appropriate agent.
"""

import json

from src.agents.llm import chat


PLANNER_SYSTEM_PROMPT = """You are a task planning specialist. Your job is to decompose a user's goal into a clear, ordered list of subtasks.

Rules:
1. Each subtask must be specific and actionable
2. Subtasks should be in logical execution order
3. Each subtask should specify which agent should handle it:
   - "researcher": For tasks that need web search or information gathering
   - "writer": For tasks that need synthesis, formatting, or report writing
4. Return ONLY valid JSON — no markdown, no explanation

Output format:
{
  "goal": "the original goal",
  "subtasks": [
    {"id": "1", "description": "...", "agent": "researcher"},
    {"id": "2", "description": "...", "agent": "researcher"},
    {"id": "3", "description": "...", "agent": "writer"}
  ]
}

Keep it to 3-6 subtasks. More than that is over-planning."""


def create_plan(goal: str) -> dict:
    """
    Take a high-level goal and return a structured plan.

    The LLM generates a JSON plan that the Orchestrator will execute
    step by step. We parse the JSON and validate it has the right structure.
    """
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Create a plan for this goal: {goal}"},
    ]

    response = chat(messages, temperature=0.1)  # Low temperature = deterministic

    # Parse the JSON response
    # LLMs sometimes wrap JSON in markdown code blocks, so strip those
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]  # Remove first line
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    try:
        plan = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: if parsing fails, create a simple default plan
        plan = {
            "goal": goal,
            "subtasks": [
                {
                    "id": "1",
                    "description": f"Research: {goal}",
                    "agent": "researcher",
                },
                {
                    "id": "2",
                    "description": f"Write a summary report about: {goal}",
                    "agent": "writer",
                },
            ],
        }

    # Validate structure
    if "subtasks" not in plan:
        plan["subtasks"] = []
    if "goal" not in plan:
        plan["goal"] = goal

    return plan
