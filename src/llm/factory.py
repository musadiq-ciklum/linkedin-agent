# src/llm/factory.py
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


def create_llm_client(provider: str) -> LLMClient:
    if provider not in LLM_PROVIDER_REGISTRY:
        available = sorted(LLM_PROVIDER_REGISTRY)
        raise ValueError(
            f"Unknown LLM provider '{provider}'. Available: {available}"
        )
    return LLM_PROVIDER_REGISTRY[provider]()
