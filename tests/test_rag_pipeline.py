# tests/test_rag_pipeline.py

from src.llm.fake import FakeLLMClient
from src.rag.pipeline import RAGPipeline
from src.prompts.prompt_builder import PromptBuilder
import src.config as config


class DummyRetriever:
    def search(self, query, top_k=5):
        return [
            {"id": "doc1", "text": "Document 1", "score": 0.4},
            {"id": "doc2", "text": "Document 2", "score": 0.3},
        ]


class DummyReranker:
    def rerank(self, query, docs):
        return docs

def test_rag_pipeline_returns_llm_output():
    retriever = DummyRetriever()
    reranker = DummyReranker()
    prompt_builder = PromptBuilder()
    llm = FakeLLMClient(answer="EXPECTED ANSWER")

    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        llm_client=llm,
        prompt_builder=prompt_builder,
    )

    result = pipeline.run("test query")

    assert result == "EXPECTED ANSWER"
