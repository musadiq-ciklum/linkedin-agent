import json

def save_chunks(chunks: list[str], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
