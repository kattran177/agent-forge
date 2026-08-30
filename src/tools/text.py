"""
Text processing tools for agents.

Simple utility functions that agents can use to manipulate text.
These aren't LLM-powered — they're deterministic Python functions
that handle the boring parts so the LLM can focus on reasoning.
"""


def summarise_text(text: str, max_chars: int = 2000) -> str:
    """
    Truncate text to fit within a character limit.

    LLMs have context windows (max input size). When feeding research
    results to the Writer agent, we may need to trim long texts.
    This is a simple truncation — for smarter summarisation,
    you'd use the LLM itself.
    """
    if len(text) <= max_chars:
        return text

    # Cut at the last sentence boundary before the limit
    truncated = text[:max_chars]
    last_period = truncated.rfind(".")
    if last_period > max_chars * 0.5:  # Only if we keep at least half
        truncated = truncated[: last_period + 1]

    return truncated + "\n\n[... truncated]"


def format_task_list(tasks: list[dict]) -> str:
    """Format a list of task dicts as a numbered list for display."""
    if not tasks:
        return "No tasks."

    lines = []
    for i, task in enumerate(tasks, 1):
        status_icon = {
            "pending": "⏳",
            "running": "🔄",
            "completed": "✅",
            "failed": "❌",
        }.get(task.get("status", "pending"), "❓")

        lines.append(f"{status_icon} {i}. {task.get('description', 'Unknown task')}")

    return "\n".join(lines)


def extract_key_points(text: str) -> list[str]:
    """
    Extract sentences that look like key points from text.

    Simple heuristic: sentences containing signal words are likely important.
    This helps agents quickly identify the most relevant information.
    """
    signal_words = [
        "important",
        "key",
        "significant",
        "notable",
        "primarily",
        "main",
        "critical",
        "essential",
        "advantage",
        "disadvantage",
        "benefit",
        "challenge",
        "feature",
        "supports",
        "enables",
        "provides",
        "designed for",
        "best for",
        "compared to",
    ]

    sentences = text.replace("\n", " ").split(".")
    key_points = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 20:
            continue
        if any(word in sentence.lower() for word in signal_words):
            key_points.append(sentence.strip() + ".")

    return key_points[:10]  # Cap at 10 points
