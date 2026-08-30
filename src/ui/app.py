"""
AgentForge Chat Interface — Streamlit UI for interacting with agents.

This provides a ChatGPT-like interface where you:
1. Type a goal in natural language
2. Watch as agents plan, research, and write in real-time
3. See the final report and full activity log
"""

import streamlit as st

from src.agents.orchestrator import run_agent
from src.memory.store import MemoryStore

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgentForge — AI Agent Platform",
    page_icon="🤖",
    layout="wide",
)

# ── Initialise session state ──────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = MemoryStore()

# ── Header ─────────────────────────────────────────────────────────
st.title("🤖 AgentForge")
st.caption(
    "Multi-agent AI system — give me a goal and I'll plan, research, and write a report"
)

# ── Sidebar: activity log ──────────────────────────────────────────
with st.sidebar:
    st.header("📋 Agent Activity")
    if "last_activity" in st.session_state:
        for entry in st.session_state.last_activity:
            agent = entry.get("agent", "unknown")
            action = entry.get("action", "")
            icon = {
                "planner": "📋",
                "researcher": "🔍",
                "writer": "✍️",
                "orchestrator": "🤖",
            }.get(agent, "⚙️")
            st.text(f"{icon} [{agent}] {action}")
    else:
        st.text("No activity yet. Submit a goal to start.")

    st.divider()
    st.caption("Powered by Groq (Llama 3.3 70B)")

# ── Chat history ───────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ─────────────────────────────────────────────────────
if prompt := st.chat_input(
    "Enter a goal (e.g., 'Research the top 3 AI agent frameworks and compare them')"
):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run the agent system
    with st.chat_message("assistant"):
        with st.spinner("🤖 Agents working..."):
            status_container = st.empty()
            status_container.text("📋 Planning tasks...")

            try:
                result = run_agent(prompt, memory=st.session_state.memory)

                # Store activity log for sidebar
                st.session_state.last_activity = result.get("activity_log", [])

                # Display the result
                final_report = result.get("final_report", "")
                findings = result.get("research_findings", "")
                plan = result.get("plan", {})
                error = result.get("error", "")

                status_container.empty()

                if error:
                    st.error(f"Agent encountered an error: {error}")
                    response_text = f"I ran into an error: {error}"
                elif final_report:
                    # Show the plan
                    with st.expander("📋 Execution Plan", expanded=False):
                        for task in plan.get("subtasks", []):
                            st.text(
                                f"  {task.get('id', '?')}. [{task.get('agent', '?')}] "
                                f"{task.get('description', '')}"
                            )

                    # Show research findings
                    if findings:
                        with st.expander("🔍 Research Findings", expanded=False):
                            st.markdown(findings)

                    # Show the final report
                    st.markdown("### 📄 Report")
                    st.markdown(final_report)
                    response_text = final_report
                else:
                    st.warning("Agents completed but produced no report.")
                    response_text = (
                        findings if findings else "No results were produced."
                    )

            except ValueError as e:
                status_container.empty()
                st.error(str(e))
                response_text = f"Configuration error: {e}"
            except Exception as e:
                status_container.empty()
                st.error(f"Unexpected error: {e}")
                response_text = f"Error: {e}"

        st.session_state.messages.append(
            {"role": "assistant", "content": response_text}
        )
