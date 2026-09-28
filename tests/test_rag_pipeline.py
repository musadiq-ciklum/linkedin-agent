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


def _make_pipeline(answer="EXPECTED ANSWER"):
    return RAGPipeline(
        retriever=DummyRetriever(),
        reranker=DummyReranker(),
        llm_client=FakeLLMClient(answer=answer),
        prompt_builder=PromptBuilder(),
    )


def test_rag_pipeline_returns_llm_output():
    result = _make_pipeline().run("test query")
    assert result == "EXPECTED ANSWER"


def test_fake_llm_stream_yields_tokens():
    llm = FakeLLMClient(answer="hello world")
    tokens = list(llm.stream("any prompt"))
    assert tokens == ["hello", "world"]


def test_stream_with_context_returns_contexts_and_full_answer():
    pipeline = _make_pipeline(answer="STREAMED ANSWER")
    contexts, token_stream = pipeline.stream_with_context("test query")
    full_text = "".join(token_stream)
    assert "STREAMED" in full_text
    assert isinstance(contexts, list)


def test_stream_with_context_returns_empty_contexts_when_no_docs_match():
    class NoResultRetriever:
        def search(self, query, top_k=5):
            return []

    pipeline = RAGPipeline(
        retriever=NoResultRetriever(),
        reranker=DummyReranker(),
        llm_client=FakeLLMClient(),
        prompt_builder=PromptBuilder(),
    )
    contexts, token_stream = pipeline.stream_with_context("anything")
    assert contexts == []
    assert "".join(token_stream) != ""
