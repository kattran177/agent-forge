"""
Agent tools — capabilities that agents can use to interact with the world.

WHAT ARE TOOLS IN AGENTIC AI?
An LLM by itself can only generate text. Tools give it the ability to ACT:
- Search the web → get real, current information
- Read files → access local data
- Run code → execute computations
- Call APIs → interact with external services

The LLM decides WHEN and HOW to use each tool based on the task.
This is the fundamental pattern behind ChatGPT plugins, Claude tools,
and every AI agent framework.

We use DuckDuckGo for web search because:
- No API key needed (free, no signup)
- No rate limits for reasonable usage
- Returns good results for research queries
"""

from dataclasses import dataclass
from duckduckgo_search import DDGS


@dataclass
class SearchResult:
    """A single web search result."""

    title: str
    url: str
    snippet: str


def web_search(query: str, max_results: int = 5) -> list[SearchResult]:
    """
    Search the web using DuckDuckGo.

    This is the primary research tool for the Researcher agent.
    It takes a natural language query and returns relevant results
    with titles, URLs, and text snippets.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("href", ""),
                snippet=r.get("body", ""),
            )
            for r in results
        ]
    except Exception as e:
        print(f"[WARN] Search failed for '{query}': {e}")
        return []


def format_search_results(results: list[SearchResult]) -> str:
    """
    Format search results as a readable string for the LLM.

    The LLM can't see raw Python objects — it needs text.
    This formats results in a way that's easy for the model to parse
    and extract information from.
    """
    if not results:
        return "No search results found."

    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(
            f"[{i}] {r.title}\n"
            f"    URL: {r.url}\n"
            f"    {r.snippet}\n"
        )

    return "\n".join(formatted)
