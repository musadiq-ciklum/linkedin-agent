# tests/test_social_post_generation.py
import re
from unittest.mock import MagicMock
from src.rag.pipeline import RAGPipeline


def sentence_count(text: str) -> int:
    # Simple sentence heuristic
    return len(re.findall(r"[.!?]", text))


def test_generate_social_post_basic_contract():
    # -------------------------
    # Arrange
    # -------------------------
    mock_llm = MagicMock()
    mock_llm.generate.return_value.text = (
        "I'm thrilled to announce the successful completion of my project, the AI Agentic RAG Assistant! "
        "This RAG-based AI agent allows users to build custom knowledge bases and ask domain-specific questions. "
        "It combines retrieval-augmented generation, agentic reasoning, tool-calling, self-reflection, and evaluation. "
        "I built this system as a solo developer, with unit tests covering most of the codebase. "
        "This project was created as part of the Ciklum AI Academy. "
        "Thanks to @Ciklum for the incredible learning experience."
    )

    pipeline = RAGPipeline(
        retriever=MagicMock(),
        reranker=None,
        llm_client=mock_llm,
        prompt_builder=MagicMock(),
    )

    # -------------------------
    # Act
    # -------------------------
    post = pipeline.generate_social_post()

    # -------------------------
    # Assert: basic structure
    # -------------------------
    assert isinstance(post, str)
    assert post.strip() != ""

    # -------------------------
    # Assert: length (LinkedIn style)
    # -------------------------
    assert 5 <= sentence_count(post) <= 8

    # -------------------------
    # Assert: required content
    # -------------------------
    lowered = post.lower()

    assert "rag" in lowered
    assert "knowledge base" in lowered or "knowledge bases" in lowered
    assert "solo" in lowered
    assert "unit test" in lowered
    assert "ciklum" in lowered
    assert "academy" in lowered

    # -------------------------
    # Assert: no templates/placeholders
    # -------------------------
    assert "[" not in post
    assert "]" not in post
    assert "fill" not in lowered
    assert "template" not in lowered

    # -------------------------
    # Assert: LLM was called
    # -------------------------
    mock_llm.generate.assert_called_once()
