# import sys
# import os

# Fix import path BEFORE importing project modules
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
from src.data_prep.pipeline import prepare_data
from src.data_prep.saver import save_chunks

input_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "sample.txt" #"data/raw/sample.txt"
output_path = "data/chunks/sample.json"

chunks = prepare_data(input_path)
save_chunks(chunks, output_path)

print("Chunks saved to:", output_path)
