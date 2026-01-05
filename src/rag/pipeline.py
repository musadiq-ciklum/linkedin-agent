# src/rag/pipeline.py
from typing import Optional, List
from src.llm.gemini import GeminiLLMClient
from src.prompts.prompt_builder import PromptBuilder
from src.rag.schema import RetrievedDoc
from src.search.reranker import BaseRanker
from src.api.schemas import AskResponse
from src.config import MIN_RELEVANCE_SCORE

class RAGPipeline:
    def __init__(
        self,
        retriever: "Retriever",
        reranker: Optional[BaseRanker],
        llm_client: GeminiLLMClient,
        prompt_builder: PromptBuilder,
        top_k: int = 5,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder
        self.top_k = top_k

    def _run_core(self, query: str, top_k: int, use_rerank: bool):
        docs = self.retriever.search(query, top_k=top_k)

        if not docs:
            return None, []

        if use_rerank and self.reranker:
            docs = self.reranker.rerank(query, docs)

        # -------------------------
        # GATING
        # -------------------------
        if docs[0].score < MIN_RELEVANCE_SCORE:
            return None, []

        top_doc = docs[0]

        # -------------------------
        # ANSWER STRATEGY
        # -------------------------
        if len(docs) == 1 or top_doc.score >= 0.65:
            # Extractive answer
            answer: str = top_doc.text
        else:
            # Generative answer
            prompt = self.prompt_builder.build(query, docs)
            llm_response = self.llm.generate(prompt)
            answer: str = llm_response.text

        return answer, docs

    # -----------------------------
    # Script-friendly
    # -----------------------------
    def run(self, query: str, top_k: Optional[int] = None, use_rerank: bool = True) -> str:
        answer, _ = self._run_core(query, top_k=top_k, use_rerank=use_rerank)
        return answer or "I could not find this information in the knowledge base."

    # -----------------------------
    # API-friendly
    # -----------------------------
    def run_with_context(
        self,
        query: str,
        top_k: int | None = None,
        use_rerank: bool = True,
    ) -> AskResponse:
        answer, docs = self._run_core(
            query=query,
            top_k=top_k or self.top_k,
            use_rerank=use_rerank,
        )

        if answer is None:
            return AskResponse(
                answer="I could not find this information in the knowledge base.",
                contexts=[],
                metadata={},
            )

        return AskResponse(
            answer=answer,
            contexts=[
                {
                    "doc_id": doc.id,
                    "score": doc.score,
                    "content": doc.text,
                }
                for doc in docs
            ],
            metadata={},
        )

