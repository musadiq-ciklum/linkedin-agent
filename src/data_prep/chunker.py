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


def chunk_text(text: str, chunk_size: int = 600) -> list[str]:
    """
    Main entry: split → detect sections → chunk.
    """
    paragraphs = split_into_paragraphs(text)
    return chunk_by_sections(paragraphs, max_size=chunk_size)
