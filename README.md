# 🤖 AI-Agentic RAG Assistant

This project is the final assignment for the **Ciklum AI Academy – Engineering Track**.  
It demonstrates an **AI-Agentic system** built on a **Retrieval-Augmented Generation (RAG)** pipeline with autonomous reasoning, tool-calling, and self-reflection.

---

## ⚡ Quick Start

Get from `git clone` to a running chat UI in 6 steps:

```bash
# 1. Clone
git clone https://github.com/musadiq-ciklum/linkedin-agent.git
cd linkedin-agent

# 2. Create virtual environment and install dependencies
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env — set GEMINI_API_KEY and AUTH_SECRET_KEY at minimum

# 4. Run database migrations
PYTHONPATH=. python scripts/migrate.py

# 5. Start the app
streamlit run app.py

# 6. Open your browser at http://localhost:8501 — register and start chatting
```

---

## 📘 Project Overview

The AI-Agentic RAG Assistant:

- 🔍 Uses a RAG pipeline for information retrieval from prepared datasets
- 🧠 Performs autonomous reasoning and self-reflection
- 🛠️ Executes tool-based actions based on reasoning outcomes
- 📊 Measures effectiveness via relevance, clarity, and accuracy metrics
- ⚙️ Supports flexible data and retrieval configuration via `.env` and `config.py`

---

## 🚀 Setup

### 📥 Clone the Repository

**SSH**
```bash
git clone git@github.com:musadiq-ciklum/linkedin-agent.git
```

**HTTPS**
```bash
git clone https://github.com/musadiq-ciklum/linkedin-agent.git
```

##  🧪 Create a Virtual Environment & Install Dependencies
```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
pip install -r requirements.txt
```


## 🔐 Configure Environment Variables
Create a copy of `.env.example` and rename it to `.env`.

**Set your Gemini key**
```
GEMINI_API_KEY=your_real_key_here
```

**Auth / Database** (required for the Streamlit app)
```
AUTH_SECRET_KEY=replace-with-a-long-random-string
SQLITE_DB_PATH=data/app.db
```

**Confluence** (optional — leave blank to disable)
```
CONFLUENCE_URL=https://your-site.atlassian.net
CONFLUENCE_USER=your-email@example.com
CONFLUENCE_API_TOKEN=your-api-token-here
CONFLUENCE_SPACE_KEY=YOUR_SPACE_KEY
```

**Git** (optional — leave blank to disable)
```
GIT_REPO_URL=https://github.com/your-org/your-repo.git
GIT_LOCAL_PATH=data/git_repos
GIT_TOKEN=your_github_token_here
```

## 🗄️ Run Database Migrations
Required once before starting the app, and after any new migrations are added:
```bash
PYTHONPATH=. python scripts/migrate.py
```

## ✅ Verify Setup
```
pytest -v
```

## 🧭 Usage

> All `python scripts/` and `python evaluation/` commands require `PYTHONPATH=.` so the `src/` package is importable. Either prefix each command or run `export PYTHONPATH=.` once in your shell.

**📚 Data Preparation**
```bash
PYTHONPATH=. python scripts/data_prep.py data/raw/sample.txt
```
Prepares documents and embeddings for the RAG pipeline.

**🗂️ Populate Vector Store (Chroma)**
```bash
PYTHONPATH=. python scripts/populate_chroma_test.py
```
Loads embeddings into a local Chroma vector database.

**🔎 Query / Search**
```bash
PYTHONPATH=. python scripts/search_test.py "Search query"
```
Performs semantic search over the vector store and returns top-k relevant documents based on configured thresholds.

**🤖 Run Agent**
```bash
PYTHONPATH=. python scripts/rag_run.py "Search query"
```
Demonstrates the AI-Agentic workflow, including retrieval, reasoning, tool-calling, self-reflection, and final response generation.

## 💬 Streamlit Web App

The project ships a browser-based chat interface built with Streamlit.

**Start the app**
```bash
streamlit run app.py
```

**Features:**
- **Login / Register** — multi-user auth with bcrypt-hashed passwords and signed session cookies
- **Persistent chat history** — every turn is saved to SQLite; previous sessions are listed in the sidebar and can be resumed, renamed, or deleted
- **Auto-named sessions** — the LLM generates a short title from your first message
- **Streaming responses** — tokens appear in real time as the LLM generates them
- **Retrieved context** — source documents shown in collapsible sidebar expanders
- **3-hour inactivity timeout** — sessions expire automatically after 3 hours of inactivity

> **Prerequisites:** run `PYTHONPATH=. python scripts/migrate.py` once before starting the app.

---

## 🌐 API Usage (FastAPI + Uvicorn)
The project exposes HTTP endpoints for querying the agent, generating embeddings, and ingesting new documents into the vector store.

**▶️ Start API Server**
```bash
uvicorn src.api.main:app --reload
```
Once running, the API will be available at:
```bash
http://127.0.0.1:8000/docs
```
Interactive API documentation is available via Swagger UI.

**POST /ask**
- Runs the full agentic RAG pipeline:
- Retrieves relevant documents from the vector store
- Performs LLM-based reasoning and optional re-ranking
- Generates a final, context-aware response

**POST /ask/stream**
- Same as `/ask` but returns a streaming `text/plain` response — tokens are yielded as they are generated by the LLM

**POST /embedding**
Generates a vector embedding for the provided input text using the configured embedding model.

**POST /upload**
Uploads a .txt or .pdf document and ingests it into the vector store:

- Extracts and cleans text
- Chunks content
- Generates embeddings
- Stores vectors for future retrieval

## 🧰 Technology Stack

This project is built using the following technologies and libraries:

- **Python 3.10+** – Core programming language
- **FastAPI** – API framework for exposing agent endpoints
- **Uvicorn** – ASGI server for running the FastAPI application
- **Streamlit** – Browser-based chat UI with auth, history, and streaming
- **Google Gemini API** – Large Language Model used for reasoning and response generation
- **Sentence-Transformers (MiniLM)** – Embedding generation for semantic retrieval
- **CrossEncoder (`ms-marco-MiniLM-L-6-v2`)** – Cross-encoder reranker for precise relevance scoring
- **ChromaDB** – Vector database for storing and retrieving embeddings
- **SQLite** – Persistent storage for user accounts and chat history
- **bcrypt + itsdangerous** – Password hashing and signed session tokens
- **FastMCP** – MCP server for exposing Confluence and Git as agent tools
- **GitPython** – Git repository cloning and file search for the Git MCP source
- **Pytest** – Unit testing and validation
- **dotenv** – Environment variable management
- **Mermaid (architecture.mmd)** – High-level system architecture visualization

The system follows a modular, agentic RAG design with configurable retrieval, reasoning, tool execution, and self-reflection components.

## 📏 Evaluation

The system includes an offline evaluation pipeline to measure retrieval quality, answer relevance, and performance characteristics of the RAG workflow.

Evaluation is performed using a small labeled dataset and compares agent behavior **with and without reranking**.

**Metrics used:**
- **Precision@K** – Measures how many retrieved documents are relevant
- **Recall@K** – Measures coverage of relevant documents
- **RAG Quality Score** – Keyword overlap between generated and expected answers
- **Latency Metrics** – Retrieval, reranking, LLM, and total response time

**Run Evaluation**
```bash
PYTHONPATH=. python evaluation/eval_rag.py
```

This script:
- Executes the RAG pipeline on a sample evaluation dataset
- Compares reranked vs non-reranked retrieval
- Exports CSV reports to:
  - evaluation/reports/report_with_rerank.csv
  - evaluation/reports/report_without_rerank.csv

The evaluation setup ensures the agent’s retrieval effectiveness, reasoning quality, and performance characteristics can be inspected and compared in a reproducible manner.


## 🔌 MCP Integrations

The agent can query live external sources using the Model Context Protocol (MCP). Each source is registered as a tool in `src/mcp/server.py` and is only activated when its credentials are present in `.env`.

### Confluence
When `CONFLUENCE_URL`, `CONFLUENCE_USER`, and `CONFLUENCE_API_TOKEN` are set, queries containing words like *"confluence"*, *"docs"*, or *"wiki"* are routed to Confluence instead of the local vector store. The agent searches the configured space and returns relevant page content.

### Git
When `GIT_REPO_URL` is set, queries containing words like *"git"*, *"repo"*, *"readme"*, or *"src/"* are routed to the Git source. The agent clones or pulls the repository on demand and searches Markdown, text, reStructuredText, and Python files for relevant content.

---

## 🛠️ Developer Skills (Claude Code)

This project ships a set of [Claude Code](https://claude.ai/code) skills under `.claude/skills/` that automate common development workflows:

| Skill | What it does |
|---|---|
| `update-readme` | Cross-checks closed GitHub issues against merged PR diffs and updates README only for verified, implemented features |
| `create-pr` | Creates a draft GitHub PR with title and body auto-generated from the current branch and diff |
| `pre-commit` | Runs the pre-commit checklist (tests, lint, branch check) and suggests a Conventional Commits message |

---

## ⚙️ Configuration
All parameters are centralized in `src/config.py`. 

Key options:

- `GEMINI_MODEL_NAME` – LLM model name (Gemini 2.5 Flash)
- `EMBEDDING_MODEL_NAME` – Embedding model
- `CHROMA_DIR` – Local vector store directory
- `MIN_RELEVANCE_SCORE` – Threshold for document relevance
- `EXTRACTIVE_SCORE_THRESHOLD` – Threshold for extractive answers
- `DEFAULT_TOP_K` – Number of documents retrieved per query
- `AUTH_SECRET_KEY` – Secret used to sign session tokens
- `SQLITE_DB_PATH` – Path to the SQLite database file (default: `data/app.db`)
- `SESSION_MAX_AGE` – Inactivity timeout in seconds (default: `10800` = 3 hours)
- `CONFLUENCE_URL / CONFLUENCE_USER / CONFLUENCE_API_TOKEN / CONFLUENCE_SPACE_KEY` – Confluence MCP credentials
- `GIT_REPO_URL / GIT_LOCAL_PATH / GIT_TOKEN / GIT_EXTENSIONS` – Git MCP configuration
