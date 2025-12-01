# src/embedders/sentence_transformers_embedder.py
from sentence_transformers import SentenceTransformer
from typing import List

class SentenceTransformerEmbedder:
    """
    Simple wrapper around SentenceTransformers.
    Automatically downloads model if missing.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        """
        return self.model.encode(texts, convert_to_numpy=True).tolist()
