# src/rag/pipeline.py
import json
from typing import Optional, List
from src.llm.gemini import GeminiLLMClient
from src.prompts.prompt_builder import PromptBuilder
from src.rag.schema import RetrievedDoc
from src.search.reranker import BaseRanker
from src.api.schemas import AskResponse
from src.config import MIN_RELEVANCE_SCORE, EXTRACTIVE_SCORE_THRESHOLD, DEFAULT_TOP_K
from src.agent.controller import AgentController

class RAGPipeline:
    def __init__(
        self,
        retriever: "Retriever",
        reranker: Optional[BaseRanker],
        llm_client: GeminiLLMClient,
        prompt_builder: PromptBuilder,
        top_k: int = DEFAULT_TOP_K,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder
        self.top_k = top_k
        self.agent = AgentController()

    def _normalize_docs(self, docs):
        normalized = []
        for d in docs:
            if isinstance(d, RetrievedDoc):
                normalized.append(d)
            else:
                normalized.append(
                    RetrievedDoc(
                        id=d["id"],
                        text=d["text"],
                        score=float(d.get("score", 1.0)),  # ← DEFAULT SCORE
                    )
                )
        return normalized

    def _run_core(self, query: str, top_k: int, use_rerank: bool):
        top_k = top_k or self.top_k
        docs = self.retriever.search(query, top_k=top_k)
        docs = self._normalize_docs(docs)

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
        if len(docs) == 1 or top_doc.score >= EXTRACTIVE_SCORE_THRESHOLD:
            # Extractive answer
            answer: str = top_doc.text
        else:
            # Generative answer
            prompt = self.prompt_builder.build(query, docs)
            llm_response = self.llm_client.generate(prompt)
            answer: str = llm_response.text

        return answer, docs

    # -----------------------------
    # Script-friendly
    # -----------------------------
    def _run_confluence(self, query: str) -> AskResponse:
        from src.mcp.client import call_tool
        try:
            raw = call_tool("search_confluence", {"query": query})
            results = json.loads(raw)
        except Exception as e:
            return AskResponse(
                answer="Confluence is not available. Please check your credentials in .env.",
                contexts=[],
                metadata={"agent_decision": "confluence", "error": str(e)},
            )

        if not results:
            return AskResponse(
                answer="No Confluence results found for your query.",
                contexts=[],
                metadata={"agent_decision": "confluence"},
            )

        docs = [RetrievedDoc(id=r["page_id"], text=r["text"], score=1.0) for r in results]

        if self.reranker:
            docs = self.reranker.rerank(query, docs)

        prompt = self.prompt_builder.build(query, docs)
        answer = self.llm_client.generate(prompt).text.strip()

        return AskResponse(
            answer=answer,
            contexts=[
                {"doc_id": r["page_id"], "score": 1.0, "content": r["text"]}
                for r in results
            ],
            metadata={"agent_decision": "confluence"},
        )

    def run(self, query: str, top_k: Optional[int] = None, use_rerank: bool = True) -> str:
        decision = self.agent.decide(query)

        if decision == "confluence":
            return self._run_confluence(query).answer

        if decision == "generate":
            return self._run_generate_only(query)

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

        decision = self.agent.decide(query)

        if decision == "confluence":
            return self._run_confluence(query)

        if decision == "generate":
            answer = self._run_generate_only(query)
            return AskResponse(
                answer=answer,
                contexts=[],
                metadata={"agent_decision": "generate"},
            )

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
    
    def _run_generate_only(self, query: str) -> str:
        """
        Handle generation-only requests (non-RAG).
        """

        # Social / announcement use case
        if "linkedin" in query.lower() or "post" in query.lower():
            return self.generate_social_post()

        # Generic generation fallback
        prompt = f"""
        Answer the following request clearly and concisely:
        {query}
        """

        llm_response = self.llm_client.generate(prompt)
        return llm_response.text.strip()

    def generate_social_post(self) -> str:
        """
        Generate a LinkedIn-style project announcement post.
        Output is ready to publish (5–7 sentences).
        """

        prompt = """
        Write a professional LinkedIn post announcing a project achievement.

        Requirements:
        - 5–6 sentences
        - Maximum 100 words total
        - Professional and concise tone (no excessive enthusiasm)
        - Ready to publish (no placeholders or templates)
        - Explain what the project does
        - Briefly mention how it was built (e.g., modular RAG pipeline, API-based design, testing)
        - Write in first person
        - Mention that it was created as part of the Ciklum AI Academy
        - Optionally mention or tag @Ciklum
        - Do NOT include emojis or exclamation marks

        Project details:
        - Project Name: AI Agentic RAG Assistant
        - Description: A RAG-based AI agent that allows users to build a custom knowledge base and ask domain-specific questions.
        - Capabilities: Retrieval-augmented generation, agentic reasoning, tool-calling, self-reflection, and evaluation.
        - Quality: Unit tests cover most of the codebase.
        - Built by: Solo developer.

        Generate ONLY the final LinkedIn post text.
        """

        llm_response = self.llm_client.generate(prompt)
        return llm_response.text.strip()
