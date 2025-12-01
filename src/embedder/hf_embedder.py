# src/embedder/hf_embedder.py
from typing import List
from sentence_transformers import SentenceTransformer
from src.embedder.base import BaseEmbedder

class HuggingFaceEmbedder(BaseEmbedder):
    """
    Local embedding using sentence-transformers, no HF account needed.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()
