# src/api/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class AskRequest(BaseModel):
    query: str
    top_k: int = 5
    use_rerank: bool = True


class RetrievedContext(BaseModel):
    doc_id: str
    score: float
    content: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    contexts: List[RetrievedContext]
    metadata: Dict

class EmbeddingResponse(BaseModel):
    embedding: List[float]
    model: str
    dimensions: int

class EmbeddingRequest(BaseModel):
    text: str = Field(..., min_length=1)
