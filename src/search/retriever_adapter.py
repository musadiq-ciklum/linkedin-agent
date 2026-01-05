# src/search/retriever_adapter.py

from src.rag.pipeline import RetrievedDoc
from src.search.semantic_search import semantic_search


class SemanticSearchRetriever:
    """
    Retriever adapter that:
    - runs vector search
    - converts results into RetrievedDoc
    - optionally applies reranking
    """

    def __init__(self, store, ranker=None):
        self.store = store
        self.ranker = ranker

    def search(self, query: str, top_k: int = 5):
        # Step 1: Vector search
        results = semantic_search(
            query=query,
            store=self.store,
            top_k=top_k,
        )

        # Step 2: Normalize to RetrievedDoc
        docs = [
            RetrievedDoc(
                text=text,
                score=score,
                id=str(i),
            )
            for i, (text, score) in enumerate(results)
        ]

        # Step 3: Optional reranking
        if self.ranker:
            docs = self.ranker.rerank(query, docs)

        return docs
