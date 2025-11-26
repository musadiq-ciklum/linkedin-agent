import re

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
