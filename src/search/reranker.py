# src/search/reranker.py
from typing import List, Tuple, Iterable
from src.config import load_gemini
from src.rag.schema import RetrievedDoc

class BaseRanker:
    def rerank(self, query: str, docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        raise NotImplementedError


class GeminiRanker(BaseRanker):
    def __init__(self, model_name: str = None):
        self.model = load_gemini(model_name) if model_name else load_gemini()

    def _score_one(self, query: str, text: str) -> float:
        prompt = f"""Rate relevance between query and document from 0 to 5. Return ONLY the number.
Query: {query}
Document: {text}
"""
        try:
            if hasattr(self.model, "generate_content"):
                response = self.model.generate_content(prompt)
                txt = getattr(response, "text", None) or response.get("text", "")
            else:
                out = self.model(prompt)
                txt = out if isinstance(out, str) else str(out)
            return float(str(txt).strip())
        except Exception:
            return 0.0

    def rerank(self, query: str, docs: List[RetrievedDoc]) -> List[RetrievedDoc]:
        for d in docs:
            d.score = self._score_one(query, d.text)

        return sorted(docs, key=lambda d: d.score, reverse=True)

class LocalSimpleRanker(BaseRanker):
    def _tokenize(self, s: str):
        import re
        return [t for t in re.findall(r"\w+", s.lower()) if t]

    def rerank(self, query: str, docs: list[RetrievedDoc]):
        qtok = set(self._tokenize(query))

        for d in docs:
            dtok = set(self._tokenize(d.text))
            inter = qtok & dtok
            d.score = float(len(inter)) / (len(qtok) or 1.0)

        return sorted(docs, key=lambda d: d.score, reverse=True)
