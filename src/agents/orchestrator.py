"""
Orchestrator — the central coordinator that manages the agent workflow.

WHAT IS LANGGRAPH?
LangGraph is a library for building agent workflows as state machines (graphs).
Instead of simple chains (A → B → C), you define:
- NODES: Steps in the workflow (plan, research, write)
- EDGES: Transitions between steps (conditional routing)
- STATE: Shared data that flows through the graph

This lets you build complex workflows with:
- Loops (research → check if enough → research more)
- Branching (if simple task → skip research)
- Parallel execution (research multiple topics simultaneously)

WHY NOT JUST USE A LOOP?
You could write this as a plain Python loop, and for simple agents that works.
But LangGraph gives you:
1. Visual debugging (you can render the graph)
2. State checkpointing (resume after crashes)
3. Human-in-the-loop (pause for user approval)
4. Streaming (stream agent progress in real-time)

These are the features that make agent systems PRODUCTION-GRADE.
"""

import uuid
from dataclasses import dataclass, field
from typing import Literal

from langgraph.graph import StateGraph, END

from src.agents.planner import create_plan
from src.agents.researcher import research
from src.agents.writer import write_report
from src.memory.store import MemoryStore, TaskRecord


# ── State Definition ───────────────────────────────────────────────
# This is the "shared memory" that flows through the graph.
# Every node can read and update this state.


@dataclass
class AgentState:
    """
    State that flows through the orchestration graph.

    Think of this as a clipboard that gets passed from agent to agent.
    Each agent reads what it needs, adds its results, and passes it on.
    """

    goal: str = ""
    plan: dict = field(default_factory=dict)
    current_task_index: int = 0
    research_findings: str = ""
    final_report: str = ""
    status: str = "idle"  # idle, planning, researching, writing, complete, error
    error: str = ""
    activity_log: list = field(default_factory=list)


# ── Node Functions ─────────────────────────────────────────────────
# Each function is a "node" in the graph — one step in the workflow.


def plan_node(state: dict) -> dict:
    """
    Node 1: Create a plan from the user's goal.

    The Planner agent decomposes the goal into subtasks.
    This determines what the rest of the workflow will do.
    """
    goal = state["goal"]
    log = state.get("activity_log", [])

    log.append({"agent": "planner", "action": "Creating execution plan..."})

    try:
        plan = create_plan(goal)
        log.append(
            {
                "agent": "planner",
                "action": f"Plan created with {len(plan.get('subtasks', []))} subtasks",
            }
        )
        return {
            "plan": plan,
            "status": "planned",
            "current_task_index": 0,
            "activity_log": log,
        }
    except Exception as e:
        log.append({"agent": "planner", "action": f"Planning failed: {e}"})
        return {"status": "error", "error": str(e), "activity_log": log}


def execute_node(state: dict) -> dict:
    """
    Node 2: Execute the current subtask.

    Looks at the current task in the plan and routes it to the right agent.
    This is where the actual work happens — research or writing.
    """
    plan = state.get("plan", {})
    subtasks = plan.get("subtasks", [])
    index = state.get("current_task_index", 0)
    findings = state.get("research_findings", "")
    log = state.get("activity_log", [])

    if index >= len(subtasks):
        return {"status": "complete", "activity_log": log}

    task = subtasks[index]
    agent_type = task.get("agent", "researcher")
    description = task.get("description", "")

    log.append(
        {
            "agent": agent_type,
            "action": f"Working on: {description}",
        }
    )

    try:
        if agent_type == "researcher":
            result = research(description, context=findings)
            # Accumulate research findings
            findings = findings + "\n\n" + result if findings else result
            log.append({"agent": "researcher", "action": "Research complete"})
            return {
                "research_findings": findings,
                "current_task_index": index + 1,
                "status": "researching",
                "activity_log": log,
            }

        elif agent_type == "writer":
            result = write_report(description, research_findings=findings)
            log.append({"agent": "writer", "action": "Report written"})
            return {
                "final_report": result,
                "current_task_index": index + 1,
                "status": "writing",
                "activity_log": log,
            }

        else:
            log.append({"agent": "unknown", "action": f"Unknown agent type: {agent_type}"})
            return {
                "current_task_index": index + 1,
                "activity_log": log,
            }

    except Exception as e:
        log.append({"agent": agent_type, "action": f"Failed: {e}"})
        return {
            "current_task_index": index + 1,
            "error": str(e),
            "activity_log": log,
        }


def should_continue(state: dict) -> Literal["execute", "end"]:
    """
    Conditional edge: decide whether to continue executing tasks or finish.

    This is the ROUTING LOGIC of the graph. After each task execution,
    we check if there are more tasks to do. If yes, loop back to execute.
    If no, we're done.

    In more complex agents, this could also:
    - Check if research quality is sufficient
    - Route to a different agent based on the task type
    - Ask for human approval before continuing
    """
    plan = state.get("plan", {})
    subtasks = plan.get("subtasks", [])
    index = state.get("current_task_index", 0)

    if state.get("status") == "error":
        return "end"

    if index < len(subtasks):
        return "execute"
    else:
        return "end"


# ── Graph Construction ─────────────────────────────────────────────


def build_graph() -> StateGraph:
    """
    Build the LangGraph workflow.

    The graph looks like this:

        START → plan → execute ←──┐
                          │       │
                          ▼       │
                     should_continue?
                      │         │
                      ▼         │
                    (more) ─────┘
                      │
                    (done) → END

    This creates a LOOP: plan once, then execute tasks one by one
    until all subtasks are complete.
    """
    # StateGraph manages a dictionary of state
    graph = StateGraph(dict)

    # Add nodes
    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)

    # Set entry point
    graph.set_entry_point("plan")

    # Add edges
    graph.add_edge("plan", "execute")  # After planning, start executing

    # Conditional edge: after execution, continue or end
    graph.add_conditional_edges(
        "execute",
        should_continue,
        {
            "execute": "execute",  # More tasks → loop back
            "end": END,  # No more tasks → finish
        },
    )

    return graph


def run_agent(goal: str, memory: MemoryStore | None = None) -> dict:
    """
    Execute the full agent workflow for a given goal.

    This is the main entry point. It:
    1. Builds the LangGraph workflow
    2. Initialises the state with the goal
    3. Runs the graph to completion
    4. Saves results to memory (if provided)
    5. Returns the final state

    Args:
        goal: The user's high-level goal
        memory: Optional MemoryStore for persistence

    Returns:
        The final state dict with plan, findings, report, and activity log
    """
    graph = build_graph()
    app = graph.compile()

    # Initial state
    initial_state = {
        "goal": goal,
        "plan": {},
        "current_task_index": 0,
        "research_findings": "",
        "final_report": "",
        "status": "starting",
        "error": "",
        "activity_log": [],
    }

    # Run the graph — this executes all nodes until END is reached
    final_state = app.invoke(initial_state)

    # Save to memory if provided
    if memory:
        task_id = str(uuid.uuid4())[:8]
        memory.save_task(
            TaskRecord(
                task_id=task_id,
                goal=goal,
                agent="orchestrator",
                input_text=goal,
                output_text=final_state.get("final_report", ""),
                status="completed" if not final_state.get("error") else "failed",
                metadata={
                    "plan": final_state.get("plan", {}),
                    "task_count": len(
                        final_state.get("plan", {}).get("subtasks", [])
                    ),
                },
            )
        )

    return final_state
