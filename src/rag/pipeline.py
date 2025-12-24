# src/rag/pipeline.py
from typing import List
from src.llm.base import LLMClient
from src.prompts.prompt_builder import PromptBuilder


class RAGPipeline:
    """
    Orchestrates end-to-end RAG flow:
    query → search → rerank → prompt → LLM → answer
    """

    def __init__(
        self,
        retriever,
        reranker,
        llm_client: LLMClient,
        prompt_builder: PromptBuilder,
        top_k: int = 5,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm_client
        self.prompt_builder = prompt_builder
        self.top_k = top_k

    def run(self, query: str) -> str:
        # 1. Retrieve
        docs = self.retriever.search(query, top_k=self.top_k)

        if not docs:
            return (
                "I couldn't find relevant information in the knowledge base "
                "to answer your question.\n"
                "Please try rephrasing the query or ensure the data has been indexed."
            )

        # 2. Rerank
        reranked_docs = self.reranker.rerank(query, docs)

        # 3. Build prompt using existing prompt builder
        prompt = self.prompt_builder.build(
            query,
            reranked_docs,
        )

        # 4. Generate answer
        response = self.llm.generate(prompt)

        # 5. Return plain text to caller
        return response.text
