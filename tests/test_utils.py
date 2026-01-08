# tests/test_utils.py
from src.utils import split_text

def test_split_text_basic():
    text = "a" * 1200
    chunks = split_text(text, size=500, overlap=50)

    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)
