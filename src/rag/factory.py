# src/rag/factory.py
from src.rag.pipeline import RAGPipeline
from src.search.retriever_adapter import SemanticSearchRetriever
from src.search.reranker import BaseRanker, LocalSimpleRanker
from src.vectorstore.db_store import ChromaStore
from src.llm.gemini import GeminiLLMClient
from src.prompts.prompt_builder import PromptBuilder
from src.config import CHROMA_DIR


def create_rag_pipeline() -> RAGPipeline:
    # 1. Shared infrastructure
    store = ChromaStore(persist_dir=CHROMA_DIR)

    # 2. Ranking
    ranker: BaseRanker = LocalSimpleRanker()

    # 3. Retriever
    retriever = SemanticSearchRetriever(
        store=store,
        ranker=ranker,
    )

    # 4. LLM + prompt
    llm_client = GeminiLLMClient()
    prompt_builder = PromptBuilder()

    # 5. Pipeline
    return RAGPipeline(
        retriever=retriever,
        reranker=ranker,
        llm_client=llm_client,
        prompt_builder=prompt_builder,
    )
