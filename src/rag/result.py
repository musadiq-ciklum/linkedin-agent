# src/rag/result.py
from dataclasses import dataclass
from typing import List


@dataclass
class RetrievedContext:
    doc_id: str
    score: float
    content: str


@dataclass
class RAGResult:
    answer: str
    contexts: List[RetrievedContext]
