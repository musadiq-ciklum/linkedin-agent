# src/vectorstore/chroma_store.py
import chromadb
import uuid
from typing import List, Dict, Any


class ChromaStore:
    """
    Wrapper for a ChromaDB persistent vector store.
    """

    def __init__(self, persist_directory: str = "data/chroma", collection_name: str = "linkedin_chunks"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, texts: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]]):
        ids = [str(uuid.uuid4()) for _ in texts]

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def search(self, query_embedding: List[float], n_results: int = 3):
        """
        Returns top N similar chunks.
        """
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
