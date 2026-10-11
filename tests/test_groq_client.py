# tests/test_groq_client.py
from unittest.mock import patch, MagicMock
import pytest
from src.llm.groq import GroqLLMClient
from src.llm.factory import create_llm_client
from src.llm.base import LLMClient, LLMResponse


def _mock_client(content: str = "hello"):
    mock_groq = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    mock_groq.chat.completions.create.return_value.choices = [choice]
    return mock_groq


def test_groq_generate_returns_llm_response():
    mock_groq = _mock_client("hello world")
    with patch("src.llm.groq.load_groq", return_value=(mock_groq, "llama-3.1-8b-instant")):
        client = GroqLLMClient()
        result = client.generate("test prompt")
    assert isinstance(result, LLMResponse)
    assert result.text == "hello world"
    assert result.model == "llama-3.1-8b-instant"


def test_groq_generate_handles_empty_content():
    mock_groq = _mock_client(None)
    with patch("src.llm.groq.load_groq", return_value=(mock_groq, "llama-3.1-8b-instant")):
        client = GroqLLMClient()
        result = client.generate("test prompt")
    assert result.text == ""


def test_groq_stream_yields_delta_tokens():
    mock_groq = MagicMock()

    def _make_chunk(text):
        chunk = MagicMock()
        chunk.choices[0].delta.content = text
        return chunk

    mock_groq.chat.completions.create.return_value = [
        _make_chunk("hello"),
        _make_chunk(" world"),
    ]

    with patch("src.llm.groq.load_groq", return_value=(mock_groq, "llama-3.1-8b-instant")):
        client = GroqLLMClient()
        tokens = list(client.stream("test prompt"))

    assert tokens == ["hello", " world"]


def test_groq_stream_skips_none_delta():
    mock_groq = MagicMock()

    def _make_chunk(text):
        chunk = MagicMock()
        chunk.choices[0].delta.content = text
        return chunk

    mock_groq.chat.completions.create.return_value = [
        _make_chunk(None),
        _make_chunk("data"),
        _make_chunk(None),
    ]

    with patch("src.llm.groq.load_groq", return_value=(mock_groq, "llama-3.1-8b-instant")):
        client = GroqLLMClient()
        tokens = list(client.stream("test prompt"))

    assert tokens == ["data"]


def test_create_llm_client_returns_groq_instance():
    with patch("src.llm.groq.load_groq", return_value=(MagicMock(), "llama-3.1-8b-instant")):
        client = create_llm_client("groq")
    assert isinstance(client, LLMClient)
    assert isinstance(client, GroqLLMClient)


def test_create_llm_client_error_lists_groq_as_available():
    with pytest.raises(ValueError, match="groq"):
        create_llm_client("nonexistent-provider")
