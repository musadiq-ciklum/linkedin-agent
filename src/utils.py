# src/utils.py
import re
import io
import pdfplumber

def clean_text(text: str) -> str:
    """
    Cleans text but preserves structure.
    - Keeps line breaks
    - Removes excessive spaces
    - Normalizes internal whitespace
    """

    # Normalize CRLF to LF
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove trailing spaces per line
    text = "\n".join(line.strip() for line in text.split("\n"))

    # Collapse multiple blank lines to max 1
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # Preserve headings / spacing — DO NOT flatten whitespace globally!
    return text.strip()

# src/utils.py

def split_text(text: str, size: int, overlap: int):
    chunks = []
    start = 0

    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks


def ingest_text(
    text: str,
    filename: str,
    embedder,
    vector_store,  # ChromaStore
    chunk_size: int = 500,
    overlap: int = 50,
):
    # 1️⃣ Split text into chunks
    chunks = split_text(text, chunk_size, overlap)

    # 2️⃣ Embed chunks
    embeddings = embedder.embed(chunks)

    # 3️⃣ Prepare lists for ChromaStore.add()
    ids = [f"{filename}_{i}" for i in range(len(chunks))]
    documents = chunks
    metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

    # 4️⃣ Add to ChromaStore
    vector_store.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(chunks)

def extract_text_from_pdf_bytes(data: bytes) -> str:
    """
    Extract text from PDF bytes and return a single cleaned string.
    """
    output = []

    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                output.append(text)

    combined = "\n\n".join(output)
    return clean_text(combined)
