# src/config.py
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

CHROMA_DIR = "data/chroma"
# RAG thresholds
MIN_RELEVANCE_SCORE = 0.3
EXTRACTIVE_SCORE_THRESHOLD = 0.65

def load_gemini(model_name="models/gemini-2.5-flash"):
    """
    Configures the Gemini API client and returns a model instance.
    """
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in .env file.")

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)
