"""
AgentForge configuration.

All settings loaded from environment variables.
Get a free Groq API key at: https://console.groq.com
"""

import os
from pathlib import Path

# ── LLM Configuration ──────────────────────────────────────────────
# Groq provides free, fast inference for open-source models.
# Llama 3.3 70B is the best open-source model as of mid-2026.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# ── Memory Configuration ──────────────────────────────────────────
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

SQLITE_PATH = str(DATA_DIR / "agent_memory.db")
CHROMA_PATH = str(DATA_DIR / "chroma_store")

# ── API Configuration ──────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
