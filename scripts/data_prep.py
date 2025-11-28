#!/usr/bin/env python3
# scripts/data-prep.py
import argparse
from src.data_prep.pipeline import prepare_data
from src.data_prep.saver import save_chunks
import os, json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input file path (txt/docx/pdf)")
    parser.add_argument("--chunk_size", type=int, default=600)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--output", default=None, help="Output json path (optional)")

    args = parser.parse_args()

    chunks = prepare_data(args.input, chunk_size=args.chunk_size, overlap=args.overlap)

    out = args.output
    if not out:
        base = os.path.splitext(os.path.basename(args.input))[0]
        out = f"data/chunks/{base}.json"

    save_chunks(chunks, out)
    print("Saved chunks:", out)

if __name__ == "__main__":
    main()
