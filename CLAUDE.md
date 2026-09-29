# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (Python 3.10+ required)
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run all tests
pytest -v

# Run a single test file
pytest tests/test_rag_pipeline.py -v

# Run database migrations (run once before starting the app, and after adding new migrations)
python scripts/migrate.py

# Start the API server
uvicorn src.api.main:app --reload

# Data pipeline: prepare raw text → chunks
python scripts/data_prep.py data/raw/sample.txt

# Populate ChromaDB vector store
python scripts/populate_chroma_test.py

# Run a semantic search query
python scripts/search_test.py "your search query"

# Run the full agentic RAG pipeline
python scripts/rag_run.py "your query"

# Run evaluation (outputs CSV reports to evaluation/reports/)
python evaluation/eval_rag.py
```

## Environment

Copy `.env.example` to `.env` and set:
- `GEMINI_API_KEY` — required for the LLM client and GeminiRanker
- `CHROMA_DIR`, `GEMINI_MODEL_NAME`, `EMBEDDING_MODEL_NAME`, `MIN_RELEVANCE_SCORE`, `EXTRACTIVE_SCORE_THRESHOLD`, `DEFAULT_TOP_K` — all have defaults in `src/config.py`

## Architecture

The system is a modular agentic RAG pipeline with these layers:

### Request flow (`POST /ask`)
1. **AgentController** (`src/agent/controller.py`) — keyword-based router: if the query triggers "generate" mode (e.g. contains "linkedin", "post", "write"), the pipeline skips retrieval and calls the LLM directly. Otherwise it proceeds to RAG retrieval.
2. **SemanticSearchRetriever** (`src/search/retriever_adapter.py`) — calls `semantic_search()` → `vector_search()` → `ChromaStore.search()`, then normalizes results to `RetrievedDoc` objects and optionally reranks.
3. **Reranker** (`src/search/reranker.py`) — two implementations: `LocalSimpleRanker` (token overlap, default) and `GeminiRanker` (LLM-scored, slower). `create_rag_pipeline()` uses `LocalSimpleRanker` for both roles.
4. **RAGPipeline** (`src/rag/pipeline.py`) — applies relevance gating (`MIN_RELEVANCE_SCORE`). Uses extractive answer (top doc text) when score ≥ `EXTRACTIVE_SCORE_THRESHOLD` or only one doc is returned; otherwise sends context to the LLM.
5. **GeminiLLMClient** (`src/llm/gemini.py`) + **PromptBuilder** (`src/prompts/prompt_builder.py`) — builds the prompt from system/user templates in `prompts/` and sends it to Gemini.

### Data ingestion flow (`POST /upload` or `scripts/data_prep.py`)
`loader → clean_text (utils.py) → chunker → saver` (into `data/chunks/`) → embed → `ChromaStore.add()`.
`ingest_text()` in `src/utils.py` covers the embed+store step when called from the API.

### Key abstractions
- **ChromaStore** (`src/vectorstore/db_store.py`) — wraps ChromaDB with a deterministic collection name (`linkedin_<model>_<hash>`) so changing the embedding model automatically creates a separate collection.
- **Embedder** — only one real implementation: `SentenceTransformerEmbedder` (MiniLM). The factory resolves string aliases like `"minilm"` via `EMBEDDING_MODEL_REGISTRY`.
- **PromptBuilder** — loads `prompts/system/<mode>.txt` and `prompts/user/<mode>.txt` at init time. Valid modes: `chat`, `summarizer`, `agent`, `refusal`, `rag_query`, `rerank_query`. Tests use `prompts_test/` directory.
- **Factory functions** — `create_rag_pipeline()` (`src/rag/factory.py`) and `create_embedder()` (`src/embedder/factory.py`) are the canonical entry points for wiring components together.

### API endpoints (`src/api/main.py`)
- `POST /ask` — full agentic RAG query, returns answer + retrieved contexts
- `POST /embedding` — generate a vector for an input string
- `POST /upload` — ingest a `.txt` or `.pdf` file into the vector store

### Evaluation (`evaluation/`)
`eval_rag.py` runs the RAG pipeline on `evaluation/data/sample_eval.json` comparing reranked vs. non-reranked results. Metrics (Precision@K, Recall@K, RAG quality score, latency) are defined in `evaluation/metrics.py`. Reports are written to `evaluation/reports/`.

## Development Rules

### Branch policy — never commit to `main`
Always create a feature branch before starting any work. Never commit or push directly to `main`.

Branch naming convention:
```
feat/<issue-number>-<short-description>   # new feature
fix/<issue-number>-<short-description>    # bug fix
docs/<issue-number>-<short-description>   # docs only
```

Examples:
```
feat/22-streamlit-gui
fix/23-auth-duplicate-user
docs/27-readme-v2
```

### Commit message format — Conventional Commits
Format: `<type>(<issue-ref>): <short description>`

| Type | When to use |
|---|---|
| `feat` | new feature |
| `fix` | bug fix |
| `docs` | documentation only |
| `test` | adding or updating tests |
| `refactor` | code change with no feature/fix |

Examples:
```
feat(#22): add Streamlit chat GUI
fix(#23): handle duplicate user registration
docs(#27): update README with v2 setup guide
test: add unit tests for auth service
refactor: extract prompt builder helper
```

Include the issue ref when the commit belongs to a tracked issue. Omit it for cross-cutting changes (`refactor:`, `test:`).

### PR title
Same format as commits. Describe the feature, not the implementation detail.

- Good: `feat(#22): add Streamlit chat GUI`
- Bad: `update app.py`

### Batch commits — no micro-commits
Only commit when a logical unit of work is complete. Do not commit partial implementations, commented-out code, or debug prints. Prefer one well-scoped commit per feature over several small ones.

### Test locally before committing
Before every commit:
```bash
pytest -v                                              # all tests must pass
PYTHONPATH=. python scripts/rag_run.py "test query"   # manual smoke test
```
Do not commit code that breaks existing tests.

### Unit tests per feature
Every new feature or bug fix must include a corresponding test in `tests/`. At minimum: one happy-path test and one edge-case test.

### Unit test naming — describe the behaviour
Test names must read as a sentence describing what the system does.

- Good: `test_user_can_register_with_valid_credentials`
- Good: `test_chat_history_loads_previous_session_on_login`
- Bad: `test_auth`, `test_1`, `test_register`
