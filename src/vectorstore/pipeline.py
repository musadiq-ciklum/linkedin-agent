# src/vectorstore/pipeline.py
import json
import glob
import os
from src.vectorstore.embeddings import EmbeddingModel
from src.vectorstore.chroma_store import ChromaStore


def index_chunks(chunks_folder: str = "data/chunks", persist_dir="data/chroma"):
    """
    Loads chunked text files and stores embeddings in ChromaDB.
    """

    model = EmbeddingModel()
    store = ChromaStore(persist_directory=persist_dir)

    chunk_files = glob.glob(os.path.join(chunks_folder, "*.json"))
    if not chunk_files:
        print("No chunk files found in:", chunks_folder)
        return

    all_texts = []
    all_metadata = []

    for file_path in chunk_files:
        with open(file_path, "r", encoding="utf-8") as f:
            chunk_data = json.load(f)

        for idx, chunk in enumerate(chunk_data["chunks"]):
            all_texts.append(chunk)
            all_metadata.append({
                "source": chunk_data.get("source", file_path),
                "chunk_index": idx
            })

    embeddings = model.embed_texts(all_texts)
    store.add_documents(all_texts, embeddings, all_metadata)

    print(f"Indexed {len(all_texts)} chunks into ChromaDB collection 'linkedin_chunks'")
