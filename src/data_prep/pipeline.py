from src.data_prep.loader import load_file
from src.utils import clean_text
from src.data_prep.chunker import chunk_text


def prepare_data(file_path: str, chunk_size: int = 500):
    raw_text = load_file(file_path)
    cleaned_text = clean_text(raw_text)
    chunks = chunk_text(cleaned_text, chunk_size=chunk_size)
    return chunks
