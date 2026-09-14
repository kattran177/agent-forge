"""
Pytest configuration — stubs unavailable third-party packages.

groq is not installed in the CI/test environment (it requires an API key
and a paid package). We stub the entire module in sys.modules before any
test file imports src.agents.llm, so the import chain succeeds without
needing the real package.

Any test that exercises actual LLM calls should mock src.agents.llm.chat
directly.
"""

import sys
import types
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Stub the `groq` package so src.agents.llm can be imported without it.
# ---------------------------------------------------------------------------

_groq_stub = types.ModuleType("groq")
_groq_stub.Groq = MagicMock  # The only name imported by src/agents/llm.py
sys.modules.setdefault("groq", _groq_stub)
