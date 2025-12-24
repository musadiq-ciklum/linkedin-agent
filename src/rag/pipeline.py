# src/rag/pipeline.py
from src.llm.base import LLMClient
from src.prompts.prompt_builder import PromptBuilder


class RAGPipeline:
    """
    Orchestrates end-to-end RAG flow:
    query → retrieve → rerank → prompt → LLM → answer
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
        # 1. Retrieve documents
        docs = self.retriever.search(query, top_k=self.top_k)

        # 2. If no relevant docs, return fallback BEFORE sending to LLM
        if not docs:
            return "I could not find this information in the knowledge base."

        # 3. Rerank documents
        docs = self.reranker.rerank(query, docs)

        # 4. Build prompt with only valid docs
        prompt = self.prompt_builder.build(query, docs)

        # 5. Generate answer from LLM (Gemini or Fake)
        response = self.llm.generate(prompt)

        # 6. Return just the text
        return response.text
