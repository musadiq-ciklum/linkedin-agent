import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# scripts/populate_chroma_test.py

from src.vectorstore.db_store import ChromaStore
import shutil
import os

PERSIST_DIR = "data/chroma_test"

# Reset the directory so the test is always clean
if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)

store = ChromaStore(
    persist_dir=PERSIST_DIR,
    embedder_name="hf",
    embedder_override="minilm"
)

docs = [
    """Python is a high-level programming language widely used in AI and data science.
    It supports libraries such as TensorFlow, PyTorch, and scikit-learn for building machine learning models.
    Developers can easily preprocess data, train models, and evaluate performance using Python.""",

    """Machine learning models require embeddings to convert text into vectors for similarity search.
    Embeddings capture semantic meaning of text and allow search engines to rank documents based on relevance.
    Proper preprocessing, normalization, and tokenization improve embedding quality.""",

    """ChromaDB is a vector database designed for fast similarity search on embeddings.
    It allows storing large collections of document vectors and supports various distance metrics like cosine similarity.
    Applications include document retrieval, question answering, and recommendation systems.""",

    """Neural networks transform text into vector representations using layers of neurons.
    These embeddings encode semantic relationships between words and sentences.
    Modern transformer-based models, such as BERT or MiniLM, produce high-quality embeddings suitable for search."""
]

ids = [f"doc_{i+1}" for i in range(len(docs))]

store.add(ids=ids, documents=docs)

print(f"Collection name: {store.collection.name}")
print(f"Number of docs: {store.collection.count()}")

# print("\nDocuments in collection:")
# results = store.collection.get(include=["documents"])
# for i, doc in enumerate(results["documents"], start=1):
#     print(f"{i}. {doc}")

