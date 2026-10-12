# tests/test_llm_router.py
import pytest
from src.llm.router import LLMRouter


@pytest.fixture
def router():
    return LLMRouter()


def test_confluence_decision_routes_to_gemini(router):
    assert router.select("what is in confluence", "confluence") == "gemini"


def test_git_decision_routes_to_gemini(router):
    assert router.select("show me the readme", "git") == "gemini"


def test_short_query_routes_to_groq(router):
    assert router.select("what is RAG", "retrieve") == "groq"


def test_long_query_routes_to_gemini(router):
    long_query = " ".join(["word"] * 101)
    assert router.select(long_query, "retrieve") == "gemini"


def test_query_at_exact_threshold_routes_to_groq(router):
    query_at_100 = " ".join(["word"] * 100)
    assert router.select(query_at_100, "retrieve") == "groq"
