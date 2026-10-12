# src/llm/factory.py
import os
from src.llm.base import LLMClient


def _make_gemini() -> LLMClient:
    from src.llm.gemini import GeminiLLMClient
    return GeminiLLMClient()


def _make_groq() -> LLMClient:
    from src.llm.groq import GroqLLMClient
    return GroqLLMClient()


LLM_PROVIDER_REGISTRY: dict[str, callable] = {
    "gemini": _make_gemini,
    "groq": _make_groq,
}

_PROVIDER_ENV_KEYS: dict[str, str] = {
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
}


def get_available_providers() -> list[str]:
    """Return registry keys whose API credentials are present in the environment."""
    return [
        name for name in LLM_PROVIDER_REGISTRY
        if os.getenv(_PROVIDER_ENV_KEYS.get(name, ""))
    ]


def create_llm_client(provider: str) -> LLMClient:
    if provider not in LLM_PROVIDER_REGISTRY:
        available = sorted(LLM_PROVIDER_REGISTRY)
        raise ValueError(
            f"Unknown LLM provider '{provider}'. Available: {available}"
        )
    return LLM_PROVIDER_REGISTRY[provider]()
