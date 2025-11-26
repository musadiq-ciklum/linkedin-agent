import re

def clean_text(text: str) -> str:
    """
    Basic text cleanup before chunking.
    Removes extra whitespace, line breaks, repeated spaces, tabs.
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    return text
