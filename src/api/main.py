# src/api/main.py
from fastapi import FastAPI
from src.api.schemas import AskRequest, AskResponse
from src.rag.factory import create_rag_pipeline

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

