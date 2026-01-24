# 🤖 AI-Agentic RAG Assistant

This project is the final assignment for the **Ciklum AI Academy – Engineering Track**.  
It demonstrates an **AI-Agentic system** built on a **Retrieval-Augmented Generation (RAG)** pipeline with autonomous reasoning, tool-calling, and self-reflection.

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

## ✅ Verify Setup
```
pytest -v
```

## 🧭 Usage

**📚 Data Preparation**
```
python scripts/data_prep.py data/raw/sample.txt
```
Prepares documents and embeddings for the RAG pipeline.

**🗂️ Populate Vector Store (Chroma)**
```
python scripts/populate_chroma_test.py
```
Loads embeddings into a local Chroma vector database.

**🔎 Query / Search**
```
python scripts/search_test.py "Search query"
```
Performs semantic search over the vector store and returns top-k relevant documents based on configured thresholds.

**🤖 Run Agent**
```
python scripts/rag/rag_run.py "Search query"
```
Demonstrates the AI-Agentic workflow, including retrieval, reasoning, tool-calling, self-reflection, and final response generation.

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

**POST /embedding**
Generates a vector embedding for the provided input text using the configured embedding model.

**POST /upload**
Uploads a .txt or .pdf document and ingests it into the vector store:

- Extracts and cleans text
- Chunks content
- Generates embeddings
- Stores vectors for future retrieval

## ⚙️ Configuration
All parameters are centralized in `src/config.py`. 

Key options:

- `GEMINI_MODEL_NAME` – LLM model name (Gemini 2.5 Flash)
- `EMBEDDING_MODEL_NAME` – Embedding model
- `CHROMA_DIR` – Local vector store directory
- `MIN_RELEVANCE_SCORE` – Threshold for document relevance
- `EXTRACTIVE_SCORE_THRESHOLD` – Threshold for extractive answers
- `DEFAULT_TOP_K` – Number of documents retrieved per query
- `MAX_CONTEXT_DOCS` – Maximum number of context documents for reasoning