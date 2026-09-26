# tests/test_reranker.py
from unittest.mock import MagicMock, patch

from src.rag.schema import RetrievedDoc
from src.search.reranker import CrossEncoderRanker, LocalSimpleRanker


def _docs(texts: list[str]) -> list[RetrievedDoc]:
    return [RetrievedDoc(id=str(i), text=t, score=1.0) for i, t in enumerate(texts)]


# ── CrossEncoderRanker ────────────────────────────────────────────────────────

def test_cross_encoder_reranker_returns_docs_sorted_by_score():
    with patch("src.search.reranker.CrossEncoder") as MockCE:
        instance = MockCE.return_value
        instance.predict.return_value = [0.2, 0.9]

        ranker = CrossEncoderRanker()
        docs = _docs(["Python is a programming language", "Machine learning uses embeddings"])
        result = ranker.rerank("What is machine learning?", docs)

        assert result[0].text == "Machine learning uses embeddings"
        assert result[1].text == "Python is a programming language"


def test_cross_encoder_reranker_updates_scores_on_docs():
    with patch("src.search.reranker.CrossEncoder") as MockCE:
        instance = MockCE.return_value
        instance.predict.return_value = [0.3, 0.8]

        ranker = CrossEncoderRanker()
        docs = _docs(["doc A", "doc B"])
        result = ranker.rerank("query", docs)

        assert result[0].score == 0.8
        assert result[1].score == 0.3


def test_cross_encoder_reranker_passes_query_doc_pairs_to_model():
    with patch("src.search.reranker.CrossEncoder") as MockCE:
        instance = MockCE.return_value
        instance.predict.return_value = [0.5]

        ranker = CrossEncoderRanker()
        docs = _docs(["some document"])
        ranker.rerank("test query", docs)

        instance.predict.assert_called_once_with([("test query", "some document")])


def test_cross_encoder_reranker_handles_single_doc():
    with patch("src.search.reranker.CrossEncoder") as MockCE:
        instance = MockCE.return_value
        instance.predict.return_value = [0.7]

        ranker = CrossEncoderRanker()
        docs = _docs(["only doc"])
        result = ranker.rerank("query", docs)

        assert len(result) == 1
        assert result[0].score == 0.7


# ── LocalSimpleRanker ─────────────────────────────────────────────────────────

def test_local_simple_ranker_ranks_by_token_overlap():
    ranker = LocalSimpleRanker()
    docs = _docs(["machine learning embeddings", "python programming language"])
    result = ranker.rerank("machine learning", docs)

    assert result[0].text == "machine learning embeddings"


def test_local_simple_ranker_returns_zero_score_for_no_overlap():
    ranker = LocalSimpleRanker()
    docs = _docs(["xyz abc def"])
    result = ranker.rerank("machine learning", docs)

    assert result[0].score == 0.0
