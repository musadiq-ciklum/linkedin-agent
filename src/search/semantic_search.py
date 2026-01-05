# src/search/semantic_search.py

from typing import List, Tuple, Optional
from src.vectorstore.db_store import ChromaStore
from .vector_search import vector_search

ScorePair = Tuple[str, float]


def semantic_search(
    query: str,
    store: Optional[ChromaStore] = None,
    top_k: int = 5,
) -> List[ScorePair]:
    """
    Perform pure vector similarity search.
    Returns: List of (text, score)
    """

    if not query or not query.strip():
        print("Empty query")
        return []

    if store is None:
        print("No Chroma store provided")
        return []

    # -------------------------
    # Vector Search
    # -------------------------
    vec_res = vector_search(query, top_k=top_k, store=store)

    if not vec_res:
        print("Vector search returned NO results.")
        return []

    # vec_res = [(text, distance_score), ...]
    return vec_res[:top_k]
