# src/data_prep/chunker.py

import re


def split_into_paragraphs(text: str) -> list[str]:
    """
    Splits text into paragraphs based on blank lines.
    Preserves structure and headings.
    """
    paragraphs = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in paragraphs if p.strip()]


def is_section_heading(paragraph: str) -> bool:
    """
    Detects headings such as:
    - UPPERCASE TITLES
    - "PHASE 1: ..."
    - lines with ---- or ****
    """
    if re.match(r"^[A-Z0-9 .:-]{6,}$", paragraph) and len(paragraph.split()) <= 10:
        return True

    if re.match(r"^PHASE\s+\d+", paragraph, re.IGNORECASE):
        return True

    return False


def chunk_by_sections(paragraphs: list[str], max_size: int = 600) -> list[str]:
    """
    Groups paragraphs into semantically meaningful section-based chunks.
    """

    chunks = []
    current_chunk = ""

    for para in paragraphs:

        # If paragraph is a heading → start new chunk
        if is_section_heading(para):
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n"
            continue

        # If adding this paragraph exceeds max_size → store and start new
        if len(current_chunk) + len(para) > max_size:
            chunks.append(current_chunk.strip())
            current_chunk = ""

        current_chunk += para + "\n\n"

    # Add last chunk if exists
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 0) -> list[str]:
    """
    Main entry: section-aware chunking with optional overlap.
    """
    paragraphs = split_into_paragraphs(text)
    base_chunks = chunk_by_sections(paragraphs, max_size=chunk_size)

    # If no overlap requested, return as-is
    if overlap <= 0:
        return base_chunks

    # Build overlapping windows of chunks
    overlapped = []
    for i in range(len(base_chunks)):
        chunk = base_chunks[i]

        # Add overlap from previous chunk
        if i > 0:
            prev_tail = base_chunks[i - 1][-overlap:]
            chunk = prev_tail + "\n" + chunk

        overlapped.append(chunk)

    return overlapped

