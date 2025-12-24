# src/embedders/factory.py
from typing import Any
from .sentence_transformer_embedder import SentenceTransformerEmbedder

MODEL_MAP = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",   # FIXED
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
}

def create_embedder(embedder_name: str = "minilm") -> Any:
    model_name = MODEL_MAP.get(embedder_name, embedder_name)
    return SentenceTransformerEmbedder(model_name=model_name)