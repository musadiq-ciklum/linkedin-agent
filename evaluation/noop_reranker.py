class NoOpReranker:
    """
    Returns documents as-is (baseline for comparison).
    """

    def rerank(self, query, docs):
        return docs
