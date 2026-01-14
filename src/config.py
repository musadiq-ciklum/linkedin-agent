# src/config.py
"""
Centralized configuration for the RAG system.

Defines all tunable parameters, model choices, and environment-dependent
values for reproducibility and demo clarity.
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env
load_dotenv()

# ============================================================
# Storage
# ============================================================

CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma")

# ============================================================
# Models
# ============================================================

# LLM
GEMINI_MODEL_NAME = os.getenv(
    "GEMINI_MODEL_NAME",
    "models/gemini-2.5-flash"
)

# Embeddings
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "minilm"   # alias resolved via embedder factory
)

# ============================================================
# RAG Thresholds
# ============================================================

MIN_RELEVANCE_SCORE = float(
    os.getenv("MIN_RELEVANCE_SCORE", 0.3)
)

EXTRACTIVE_SCORE_THRESHOLD = float(
    os.getenv("EXTRACTIVE_SCORE_THRESHOLD", 0.65)
)

# ============================================================
# Retrieval Defaults
# ============================================================

DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 5))
MAX_CONTEXT_DOCS = int(os.getenv("MAX_CONTEXT_DOCS", 8))

# ============================================================
# Gemini Loader
# ============================================================

def load_gemini(model_name: str | None = None):
    """
    Configures the Gemini API client and returns a model instance.
    """
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in .env file.")

    genai.configure(api_key=api_key)

    return genai.GenerativeModel(
        model_name or GEMINI_MODEL_NAME
    )
