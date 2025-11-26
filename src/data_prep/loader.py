import os
import docx
import pypdf

def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_docx(path: str) -> str:
    doc = docx.Document(path)
    return "\n".join([p.text for p in doc.paragraphs])


def load_pdf(path: str) -> str:
    text = ""
    with open(path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def load_file(path: str) -> str:
    """
    Smart loader that detects extension and uses correct parser.
    """
    extension = os.path.splitext(path)[1].lower()

    if extension == ".txt":
        return load_txt(path)
    elif extension == ".docx":
        return load_docx(path)
    elif extension == ".pdf":
        return load_pdf(path)
    else:
        raise ValueError(f"Unsupported file format: {extension}")
