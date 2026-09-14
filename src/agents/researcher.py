"""
Researcher Agent — gathers information from the web.

This agent is a ReAct (Reasoning + Acting) agent:
1. THINK: What information do I need?
2. ACT: Search the web for it
3. OBSERVE: Read the results
4. THINK: Do I have enough info, or do I need another search?
5. RESPOND: Return the gathered information

The ReAct pattern is one of the most important concepts in agentic AI.
It was introduced by Yao et al. (2022) and is used by virtually every
agent framework.
"""

from src.agents.llm import chat
from src.core.logging import get_logger
from src.tools.search import web_search, format_search_results

_logger = get_logger("researcher")


RESEARCHER_SYSTEM_PROMPT = """You are a research specialist. Your job is to gather accurate, relevant information for a given task.

You have access to web search results. Based on these results:
1. Extract the most relevant facts and details
2. Note the sources (URLs) for important claims
3. Identify any gaps — what's missing that we still need?
4. Be specific with data, names, dates, and comparisons

Format your findings clearly with bullet points. Include source URLs.
If the search results don't fully answer the question, say what's missing."""


def research(task_description: str, context: str = "") -> str:
    """
    Execute a research task by searching the web and synthesising results.

    Args:
        task_description: What to research (from the Planner's subtask)
        context: Previous research findings to build upon

    Returns:
        Structured research findings as text
    """
    _logger.info("Starting research", task=task_description)

    # Step 1: Search the web
    results = web_search(task_description, max_results=5)
    formatted_results = format_search_results(results)
    _logger.info("Web search complete", result_count=len(results))

    # Step 2: Ask the LLM to synthesise the search results
    context_block = ""
    if context:
        context_block = f"\n\nPrevious research findings:\n{context}\n"

    messages = [
        {"role": "system", "content": RESEARCHER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Research task: {task_description}\n\n"
                f"Web search results:\n{formatted_results}"
                f"{context_block}\n\n"
                f"Synthesise the most relevant information for this task."
            ),
        },
    ]

    findings = chat(messages, temperature=0.2)
    _logger.info("Research synthesis complete", task=task_description)
    return findings
