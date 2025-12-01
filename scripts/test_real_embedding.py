# scripts/test_real_embedding.py
import os
os.environ["POSTHOG_DISABLE"] = "1"
from src.vectorstore.db_store import ChromaStore

# Use your real ONNX embedder
store = ChromaStore(persist_dir="data/chroma_test", embedder_override="hf")

# Example text
text = "Hello world! This is a test of embeddings."

# Use the embedder directly
vec = store.embedder.embed([text])[0]

print(f"Text: {text}")
print(f"Vector length: {len(vec)}")
print(f"First 10 vector values: {vec[:10]}")
