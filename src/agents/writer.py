"""
Writer Agent — synthesises research findings into structured reports.

This agent takes all the information gathered by the Researcher and
produces a well-formatted, cohesive output. It's the final step in
the pipeline — turning raw research into a deliverable.

The separation between Researcher and Writer is important:
- Researcher optimises for BREADTH (gathering many data points)
- Writer optimises for DEPTH and CLARITY (making sense of it all)

This is a common pattern in multi-agent systems: specialised agents
do better than one generalist agent trying to do everything.
"""

from src.agents.llm import chat


WRITER_SYSTEM_PROMPT = """You are a report writing specialist. Your job is to synthesise research findings into a clear, well-structured report.

Guidelines:
1. Start with an executive summary (2-3 sentences)
2. Organise information with clear headings
3. Use bullet points for comparisons and lists
4. Include specific details, numbers, and examples
5. End with key takeaways or recommendations
6. Cite sources when available

Write in a professional but accessible tone. Assume the reader is technically literate but may not be an expert in this specific topic."""


def write_report(task_description: str, research_findings: str) -> str:
    """
    Write a structured report based on research findings.

    Args:
        task_description: What to write about (from the Planner's subtask)
        research_findings: Accumulated research from the Researcher agent

    Returns:
        A formatted report as text (Markdown)
    """
    messages = [
        {"role": "system", "content": WRITER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Writing task: {task_description}\n\n"
                f"Research findings to synthesise:\n{research_findings}\n\n"
                f"Write a comprehensive, well-structured report."
            ),
        },
    ]

    return chat(messages, temperature=0.3)  # Slightly more creative for writing
