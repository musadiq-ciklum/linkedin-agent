# src/embedders/factory.py
from typing import Any
from .sentence_transformers_embedder import SentenceTransformerEmbedder
from .fake import FakeEmbedder

def create_embedder(embedder_name: str = "hf") -> Any:
    """
    Factory to create an embedder instance.
    Default uses SentenceTransformers for pip-only install.
    """
    embedder_name = embedder_name.lower()
    
    if embedder_name in ["hf", "sentence-transformers", "st"]:
        return SentenceTransformerEmbedder()
    elif embedder_name == "fake":
        return FakeEmbedder()
    else:
        raise ValueError(f"Unsupported embedder: {embedder_name}")
