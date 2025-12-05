# tests/test_chroma_store.py
import tempfile
from src.vectorstore.db_store import ChromaStore

def test_chromastore_initialization():
    # Temporary isolated DB
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ChromaStore(persist_dir=tmpdir, embedder_override="minilm")

    # Ensure embedder is SentenceTransformerEmbedder
    from src.embedder.sentence_transformer_embedder import SentenceTransformerEmbedder
    assert isinstance(store.embedder, SentenceTransformerEmbedder)
