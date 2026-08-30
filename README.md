# 🤖 AgentForge — Multi-Agent AI Task Orchestration Platform

A multi-agent system that takes a high-level goal, decomposes it into tasks, and orchestrates specialized AI agents to execute them autonomously. Built with LangGraph, Groq (Llama 3.3), FastAPI, and Streamlit.

**Why this exists:** Agentic AI is the next paradigm shift after chatbots. This project demonstrates production-grade agent orchestration — the skill every AI company will need in 2027-2035.

---

## 🏗️ Architecture

```
                          ┌─────────────────────┐
                          │   User Interface     │
                          │   (Streamlit Chat)   │
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
                          │   FastAPI Gateway     │
                          │   /api/v1/tasks       │
                          └──────────┬───────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │       Orchestrator Agent         │
                    │  (LangGraph State Machine)       │
                    │                                  │
                    │  Decides which agent to invoke    │
                    │  next based on task + state       │
                    └──┬────────────┬────────────┬─────┘
                       │            │            │
              ┌────────▼──┐  ┌─────▼─────┐  ┌───▼────────┐
              │  Planner  │  │ Researcher│  │  Writer    │
              │  Agent    │  │  Agent    │  │  Agent     │
              │           │  │           │  │            │
              │ Breaks    │  │ Searches  │  │ Synthesises│
              │ goal into │  │ web for   │  │ findings   │
              │ subtasks  │  │ info      │  │ into       │
              │           │  │           │  │ reports    │
              └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
                    │              │              │
                    ▼              ▼              ▼
              ┌──────────────────────────────────────┐
              │          Shared Memory               │
              │  (SQLite history + ChromaDB vectors) │
              └──────────────────────────────────────┘
```

**How it works:**
1. User submits a goal (e.g., "Research the latest AI agent frameworks and write a comparison")
2. **Orchestrator** manages the workflow as a LangGraph state machine
3. **Planner** decomposes the goal into ordered subtasks
4. **Researcher** uses web search tools to gather information
5. **Writer** synthesises findings into a structured report
6. Each agent's output is stored in shared **memory** for context

---

## ⚙️ Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Agent Framework | LangGraph | Graph-based agent orchestration with state machines |
| LLM | Groq (Llama 3.3 70B) | Free tier, blazing fast inference (~500 tok/s) |
| API | FastAPI | Async Python API with auto-generated docs |
| Memory | SQLite + ChromaDB | Conversation history + semantic vector search |
| Tools | Custom Python | Web search (DuckDuckGo), text processing |
| Frontend | Streamlit | Chat interface with real-time agent activity |
| Containers | Docker Compose | One-command deployment |
| CI/CD | GitHub Actions | Automated testing |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- A free Groq API key (get one at https://console.groq.com)

### Setup

```bash
# Clone and enter project
git clone https://github.com/kattran177/agent-forge.git
cd agent-forge

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Set your API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Run the chat interface
streamlit run src/ui/app.py
```

### Run with Docker

```bash
# Set your API key in .env first
docker compose up --build
# Chat UI: http://localhost:8501
# API docs: http://localhost:8000/docs
```

---

## 💬 Example Usage

**Goal:** "Research the top 3 AI agent frameworks in 2026 and compare their strengths"

**What AgentForge does:**
1. 📋 **Planner** creates subtasks: search for frameworks → gather details on each → compare features → write report
2. 🔍 **Researcher** searches the web for each framework, extracts key information
3. ✍️ **Writer** synthesises findings into a structured comparison with pros/cons
4. 📄 **Result** delivered as a formatted report with sources

---

## 📁 Project Structure

```
agent-forge/
├── src/
│   ├── agents/             # Agent definitions
│   │   ├── orchestrator.py # LangGraph workflow — the brain
│   │   ├── planner.py      # Goal decomposition agent
│   │   ├── researcher.py   # Web research agent
│   │   └── writer.py       # Report writing agent
│   ├── tools/              # Agent tools (capabilities)
│   │   ├── search.py       # DuckDuckGo web search
│   │   └── text.py         # Text processing utilities
│   ├── memory/             # Shared agent memory
│   │   ├── store.py        # SQLite conversation history
│   │   └── vectors.py      # ChromaDB vector store
│   ├── api/                # FastAPI REST API
│   │   └── main.py         # API endpoints
│   └── ui/                 # Streamlit chat interface
│       └── app.py          # Dashboard
├── tests/                  # Test suite
├── infra/docker/           # Dockerfiles
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🧪 Testing

```bash
pytest tests/ -v --cov=src
```

---

## 🔑 Key Concepts Demonstrated

- **Agentic AI**: Autonomous systems that plan, act, and reflect
- **LangGraph State Machines**: Graph-based workflow orchestration (conditional edges, cycles)
- **Tool Use**: Agents calling external APIs and processing results
- **Shared Memory**: Conversation history + vector similarity search for context retrieval
- **Async Processing**: FastAPI async endpoints for non-blocking agent execution

---

## 📈 What I Learned

- How LLMs become agents through tool use and planning loops
- Graph-based orchestration vs simple chains (why LangGraph over LangChain)
- The challenge of agent reliability — retries, fallbacks, and guardrails
- Vector databases for semantic memory retrieval
- Designing APIs for long-running async tasks

---

## License

MIT
