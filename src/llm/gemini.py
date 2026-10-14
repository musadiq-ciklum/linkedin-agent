# src/llm/gemini.py
import os
from typing import Generator
import google.generativeai as genai
from src.llm.base import LLMClient, LLMResponse
from src.config import GEMINI_MODEL_NAME


def load_gemini(model_name: str | None = None):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in .env file.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name or GEMINI_MODEL_NAME)


class GeminiLLMClient(LLMClient):

    def __init__(self, model_name: str = "models/gemini-2.5-flash"):
        self.model_name = model_name
        self.model = load_gemini(model_name)

    def generate(self, prompt: str) -> LLMResponse:
        response = self.model.generate_content(prompt)

        # Gemini may return no parts → response.text raises ValueError
        try:
            text = response.text
        except Exception:
            text = "I could not find this information in the knowledge base."

        return LLMResponse(
            text=text,
            model=self.model_name,
            usage=None,
        )

    def stream(self, prompt: str) -> Generator[str, None, None]:
        response = self.model.generate_content(prompt, stream=True)
        for chunk in response:
            try:
                if chunk.text:
                    yield chunk.text
            except Exception:
                pass
