# src/rag/pipeline.py
import json
from typing import Optional, List, Generator, Tuple
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

    def _run_git(self, query: str) -> AskResponse:
        from src.mcp.client import call_tool
        try:
            raw = call_tool("search_git", {"query": query})
            results = json.loads(raw)
        except Exception as e:
            return AskResponse(
                answer="Git source is not available. Please check GIT_REPO_URL in .env.",
                contexts=[],
                metadata={"agent_decision": "git", "error": str(e)},
            )

        if not results:
            return AskResponse(
                answer="No Git repository results found for your query.",
                contexts=[],
                metadata={"agent_decision": "git"},
            )

        docs = [RetrievedDoc(id=r["doc_id"], text=r["text"], score=1.0) for r in results]

        if self.reranker:
            docs = self.reranker.rerank(query, docs)

        prompt = self.prompt_builder.build(query, docs)
        answer = self.llm_client.generate(prompt).text.strip()

        return AskResponse(
            answer=answer,
            contexts=[
                {"doc_id": r["doc_id"], "score": 1.0, "content": r["text"]}
                for r in results
            ],
            metadata={"agent_decision": "git"},
        )

    def run(self, query: str, top_k: Optional[int] = None, use_rerank: bool = True) -> str:
        decision = self.agent.decide(query)

        if decision == "confluence":
            return self._run_confluence(query).answer

        if decision == "git":
            return self._run_git(query).answer

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

        if decision == "git":
            return self._run_git(query)

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
        if "linkedin" in query.lower() or "post" in query.lower():
            return self.generate_social_post()
        prompt = f"Answer the following request clearly and concisely:\n{query}"
        return self.llm_client.generate(prompt).text.strip()

    # -----------------------------
    # Streaming
    # -----------------------------
    def stream_with_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_rerank: bool = True,
    ) -> Tuple[List[dict], Generator[str, None, None]]:
        """
        Returns (contexts, token_stream).
        Contexts are resolved synchronously; tokens stream from the LLM.
        """
        decision = self.agent.decide(query)

        if decision == "confluence":
            return self._stream_confluence(query)
        if decision == "git":
            return self._stream_git(query)
        if decision == "generate":
            return self._stream_generate_only(query)

        return self._stream_core(query, top_k=top_k or self.top_k, use_rerank=use_rerank)

    def _stream_core(
        self, query: str, top_k: int, use_rerank: bool
    ) -> Tuple[List[dict], Generator[str, None, None]]:
        docs = self.retriever.search(query, top_k=top_k)
        docs = self._normalize_docs(docs)

        if not docs or docs[0].score < MIN_RELEVANCE_SCORE:
            return [], iter(["I could not find this information in the knowledge base."])

        if use_rerank and self.reranker:
            docs = self.reranker.rerank(query, docs)

        contexts = [{"doc_id": d.id, "score": d.score, "content": d.text} for d in docs]
        top_doc = docs[0]

        if len(docs) == 1 or top_doc.score >= EXTRACTIVE_SCORE_THRESHOLD:
            return contexts, iter([top_doc.text])

        prompt = self.prompt_builder.build(query, docs)
        return contexts, self.llm_client.stream(prompt)

    def _stream_confluence(self, query: str) -> Tuple[List[dict], Generator[str, None, None]]:
        from src.mcp.client import call_tool
        try:
            raw = call_tool("search_confluence", {"query": query})
            results = json.loads(raw)
        except Exception as e:
            return [], iter([f"Confluence is not available. Please check your credentials in .env."])

        if not results:
            return [], iter(["No Confluence results found for your query."])

        docs = [RetrievedDoc(id=r["page_id"], text=r["text"], score=1.0) for r in results]
        if self.reranker:
            docs = self.reranker.rerank(query, docs)

        contexts = [{"doc_id": r["page_id"], "score": 1.0, "content": r["text"]} for r in results]
        prompt = self.prompt_builder.build(query, docs)
        return contexts, self.llm_client.stream(prompt)

    def _stream_git(self, query: str) -> Tuple[List[dict], Generator[str, None, None]]:
        from src.mcp.client import call_tool
        try:
            raw = call_tool("search_git", {"query": query})
            results = json.loads(raw)
        except Exception as e:
            return [], iter(["Git source is not available. Please check GIT_REPO_URL in .env."])

        if not results:
            return [], iter(["No Git repository results found for your query."])

        docs = [RetrievedDoc(id=r["doc_id"], text=r["text"], score=1.0) for r in results]
        if self.reranker:
            docs = self.reranker.rerank(query, docs)

        contexts = [{"doc_id": r["doc_id"], "score": 1.0, "content": r["text"]} for r in results]
        prompt = self.prompt_builder.build(query, docs)
        return contexts, self.llm_client.stream(prompt)

    def _stream_generate_only(self, query: str) -> Tuple[List[dict], Generator[str, None, None]]:
        if "linkedin" in query.lower() or "post" in query.lower():
            prompt = self._social_post_prompt()
        else:
            prompt = f"Answer the following request clearly and concisely:\n{query}"
        return [], self.llm_client.stream(prompt)

    def _social_post_prompt(self) -> str:
        return """
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

    def generate_social_post(self) -> str:
        llm_response = self.llm_client.generate(self._social_post_prompt())
        return llm_response.text.strip()
