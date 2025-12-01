# src/embedder/fake.py
import random
from .base import BaseEmbedder
from typing import List

class FakeEmbedder(BaseEmbedder):
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Deterministic fake embedder for unit tests.
        Returns 384-dim vectors.
        """
        vectors = []
        for text in texts:
            random.seed(hash(text) % 10_000)  # deterministic per text
            vectors.append([random.random() for _ in range(384)])
        return vectors
