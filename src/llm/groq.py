# src/llm/groq.py
import os
from typing import Generator
from src.llm.base import LLMClient, LLMResponse
from src.config import GROQ_MODEL_NAME


def load_groq(model_name: str | None = None):
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY in .env file.")
    return Groq(api_key=api_key), model_name or GROQ_MODEL_NAME


class GroqLLMClient(LLMClient):

    def __init__(self, model_name: str | None = None):
        self._client, self.model_name = load_groq(model_name)

    def generate(self, prompt: str) -> LLMResponse:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content or ""
        return LLMResponse(text=text, model=self.model_name)

    def stream(self, prompt: str) -> Generator[str, None, None]:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
