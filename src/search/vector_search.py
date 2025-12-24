# src/search/vector_search.py
from typing import List, Tuple, Optional
from src.vectorstore.db_store import ChromaStore
from src.config import CHROMA_DIR

_store: Optional[ChromaStore] = None


def get_store(persist_dir: str = CHROMA_DIR) -> ChromaStore:
    global _store
    if _store is None:
        _store = ChromaStore(persist_dir=persist_dir)
    return _store


def _distance_to_similarity(distance: float) -> float:
    try:
        d = float(distance)
    except Exception:
        return 0.0
    return 1.0 / (1.0 + max(0.0, d))


def vector_search(
    query: str,
    top_k: int = 5,
    store: Optional[ChromaStore] = None,
    persist_dir: str = CHROMA_DIR
) -> List[Tuple[str, float]]:
    
    if not query or not query.strip():
        return []

    # -----------------------------
    # Ensure we have a store
    # -----------------------------
    if store is None:
        store = get_store(persist_dir=persist_dir)

    # -----------------------------
    # Embed the query properly
    # -----------------------------
    try:
        query_vec = store.embedder.embed([query])  # <-- FIXED
    except Exception:
        return []

    # -----------------------------
    # Run vector search by embeddings
    # -----------------------------
    try:
        result = store.collection.query(
            query_embeddings=query_vec,
            n_results=top_k,
            include=["documents", "distances"]
        )
    except Exception:
        return []

    # -----------------------------
    # Extract results
    # -----------------------------
    documents = []
    distances = []

    docs_lists = result.get("documents")
    dist_lists = result.get("distances")

    if docs_lists and isinstance(docs_lists, list):
        documents = docs_lists[0] or []

    if dist_lists and isinstance(dist_lists, list):
        distances = dist_lists[0] or []

    if not documents:
        return []

    # -----------------------------
    # Convert distance → similarity
    # -----------------------------
    out = []
    for doc, dist in zip(documents, distances):
        sim = _distance_to_similarity(dist)
        out.append((doc, sim))

    return sorted(out, key=lambda x: x[1], reverse=True)[:top_k]
