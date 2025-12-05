# src/search/semantic_search.py

from typing import List, Tuple, Optional
from src.vectorstore.db_store import ChromaStore
from .vector_search import vector_search
from .reranker import BaseRanker, LocalSimpleRanker

ScorePair = Tuple[str, float]


def semantic_search(
    query: str,
    store: Optional[ChromaStore] = None,
    top_k: int = 5,
    ranker: Optional[BaseRanker] = None,
) -> List[ScorePair]:

    if not query or not query.strip():
        print("Empty query")
        return []

    if store is None:
        print("No Chroma store provided")
        return []

    # -------------------------
    # Step 1: Vector Search
    # -------------------------
    vec_res = vector_search(query, top_k=top_k, store=store)

    if not vec_res:
        print("Vector search returned NO results.")
        return []

    # vec_res = [(doc, distance_score), ...]
    docs_only = [text for text, score in vec_res]

    # -------------------------
    # Step 2: Rerank
    # -------------------------
    if ranker is None:
        ranker = LocalSimpleRanker()

    reranked = ranker.score(query, docs_only)  # returns [(doc, score)]

    # -------------------------
    # Step 3: Sorted output
    # -------------------------
    final = sorted(reranked, key=lambda x: x[1], reverse=True)

    return final[:top_k]
