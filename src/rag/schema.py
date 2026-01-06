# src/rag/schema.py
from dataclasses import dataclass

@dataclass
class Chunk:
    id: str
    text: str
    score: float = 0.0

@dataclass
class RetrievedDoc:
    id: str
    text: str
    score: float = 0.0
