# src/embedders/factory.py
from typing import Any
from .sentence_transformer_embedder import SentenceTransformerEmbedder
from .model_registry import EMBEDDING_MODEL_REGISTRY

def create_embedder(embedder_name: str) -> Any:
    model_name = EMBEDDING_MODEL_REGISTRY.get(embedder_name)
    if not model_name:
        raise ValueError(f"Unsupported embedding model: {embedder_name}")

    return SentenceTransformerEmbedder(model_name=model_name)
