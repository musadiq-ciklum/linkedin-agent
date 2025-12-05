# src/search/reranker.py
from typing import List, Tuple, Iterable
from src.config import load_gemini

Score = Tuple[str, float]


class BaseRanker:
    def score(self, query: str, texts: Iterable[str]) -> List[Score]:
        raise NotImplementedError()


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

    def score(self, query: str, texts: Iterable[str]) -> List[Score]:
        scored = []
        for t in texts:
            s = self._score_one(query, t)
            scored.append((t, float(s)))
        return sorted(scored, key=lambda x: x[1], reverse=True)


class LocalSimpleRanker(BaseRanker):
    def _tokenize(self, s: str):
        import re
        return [t for t in re.findall(r"\w+", s.lower()) if t]

    def score(self, query: str, texts: Iterable[str]) -> List[Score]:
        qtok = set(self._tokenize(query))
        scored = []
        for t in texts:
            dtok = set(self._tokenize(t))
            inter = qtok & dtok
            score = float(len(inter)) / (len(qtok) if qtok else 1.0)
            scored.append((t, score))
        return sorted(scored, key=lambda x: x[1], reverse=True)
