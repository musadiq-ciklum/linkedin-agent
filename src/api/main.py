# src/api/main.py
from fastapi import FastAPI
from src.api.schemas import AskRequest, AskResponse
from src.rag.factory import create_rag_pipeline
from src.embedder.factory import create_embedder
from src.api.schemas import EmbeddingRequest, EmbeddingResponse

app = FastAPI(title="RAG API")

# Initialize pipeline ONCE
rag_pipeline = create_rag_pipeline()

@app.get("/")
def root():
    return {"message": "LinkedIn RAG API is running"}

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    return rag_pipeline.run_with_context(
        query=request.query,
        top_k=request.top_k,
        use_rerank=request.use_rerank,
    )

embedder = create_embedder()  # singleton, model loads once

@app.post("/embedding", response_model=EmbeddingResponse)
def embedding(req: EmbeddingRequest):
    vectors = embedder.embed([req.text])  # batch
    vector = vectors[0]                   # single embedding

    return EmbeddingResponse(
        embedding=vector,
        dimensions=len(vector),
        model=embedder.model_name,
    )
