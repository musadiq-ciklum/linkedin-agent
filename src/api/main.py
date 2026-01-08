# src/api/main.py
from fastapi import FastAPI, UploadFile, File
from src.api.schemas import AskRequest, AskResponse
from src.rag.factory import create_rag_pipeline
from src.embedder.factory import create_embedder
from src.api.schemas import EmbeddingRequest, EmbeddingResponse, UploadResponse
from src.utils import ingest_text

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

@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    text = (await file.read()).decode("utf-8")

    chunks_created = ingest_text(
        text=text,
        filename=file.filename,
        embedder=embedder,
        vector_store=rag_pipeline.retriever.store,  # ChromaStore
    )

    return UploadResponse(
        status="success",
        filename=file.filename,
        chunks_created=chunks_created,
        collection="default",
    )
