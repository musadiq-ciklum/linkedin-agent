# tests/test_llm_factory.py
import os
from unittest.mock import patch, MagicMock
import pytest
from src.llm.factory import create_llm_client, get_available_providers
from src.llm.base import LLMClient


def test_create_llm_client_raises_for_unknown_provider():
    with pytest.raises(ValueError, match="Unknown LLM provider 'unknown-provider'"):
        create_llm_client("unknown-provider")


def test_create_llm_client_error_lists_available_providers():
    with pytest.raises(ValueError, match="gemini"):
        create_llm_client("nonexistent")


def test_create_llm_client_returns_llm_client_instance():
    with patch("src.llm.gemini.load_gemini", return_value=MagicMock()):
        client = create_llm_client("gemini")
    assert isinstance(client, LLMClient)


def test_get_available_providers_returns_configured_providers():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "key", "GROQ_API_KEY": ""}):
        providers = get_available_providers()
    assert "gemini" in providers
    assert "groq" not in providers


def test_get_available_providers_returns_empty_when_no_keys_set():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GROQ_API_KEY": ""}):
        providers = get_available_providers()
    assert providers == []
