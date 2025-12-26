# tests/test_eval_metrics.py
from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    keyword_overlap_score,
)


def test_precision_at_k():
    retrieved = ["d1", "d2", "d3"]
    relevant = ["d2"]
    assert precision_at_k(retrieved, relevant, 3) == 1 / 3


def test_recall_at_k():
    retrieved = ["d1", "d2"]
    relevant = ["d2", "d3"]
    assert recall_at_k(retrieved, relevant, 2) == 0.5


def test_keyword_overlap():
    generated = "retrieval augmented generation uses retrieval"
    reference = "retrieval augmented generation combines retrieval with llms"
    score = keyword_overlap_score(generated, reference)
    assert score > 0.3
