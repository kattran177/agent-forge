"""
LLM client wrapper — handles all communication with Groq.

WHY A WRAPPER?
Instead of calling the Groq API directly in every agent, we centralise it here.
This means:
1. One place to change if we switch LLM providers (Groq → OpenAI → Anthropic)
2. Consistent error handling and retries
3. Easy to mock in tests

GROQ vs OPENAI vs ANTHROPIC:
- Groq: Free tier, extremely fast (500+ tokens/sec), runs Llama/Mixtral
- OpenAI: Most capable (GPT-4), but costs money
- Anthropic: Best for safety/reasoning (Claude), but costs money

For a portfolio project, Groq is perfect — free, fast, and impressive.
"""

from groq import Groq

from src.config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE


def get_client() -> Groq:
    """Create a Groq client. Raises ValueError if API key is missing."""
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com"
        )
    return Groq(api_key=GROQ_API_KEY)


def chat(
    messages: list[dict],
    model: str = LLM_MODEL,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = 4096,
) -> str:
    """
    Send a chat completion request to Groq and return the response text.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
                  Roles: 'system' (instructions), 'user' (input), 'assistant' (previous response)
        model: Which model to use
        temperature: Creativity (0.0 = deterministic, 1.0 = creative)
        max_tokens: Maximum response length

    Returns:
        The model's response text.

    Example:
        response = chat([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Python?"}
        ])
    """
    client = get_client()

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return completion.choices[0].message.content or ""
