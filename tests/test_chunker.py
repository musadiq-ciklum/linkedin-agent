# tests/test_chunker.py
from src.data_prep.chunker import chunk_text

def test_chunking_basic():
    text = "Header\n\nParagraph one line.\n\nPHASE 1: DATA COLLECTION\n• A\n• B\n\nLong paragraph " + ("x"*700)
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) >= 2
    # ensure headings preserved
    assert any("PHASE 1" in c for c in chunks)
