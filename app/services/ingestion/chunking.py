"""
Ingestion — chunking layer.

Reads PDF, DOCX, TXT, and Markdown files, splits them into
overlapping chunks suitable for embedding.
"""

from pathlib import Path
from pypdf import PdfReader
from docx import Document
import re


# ── File readers ──────────────────────────────────────────────────

def read_pdf(document):
    reader = PdfReader(document)
    return "\n".join((p.extract_text() or "") for p in reader.pages)


def read_docx(document):
    reader = Document(document)
    return "\n".join([paragraph.text for paragraph in reader.paragraphs])


def read_txt(document):
    with open(document, encoding="utf8", errors="ignore") as f:
        return f.read()


def read_md(document):
    return Path(document).read_text(encoding="utf8", errors="ignore")


FORMAT_HANDLER = {
    ".pdf": read_pdf,
    ".docx": read_docx,
    ".txt": read_txt,
    ".md": read_md,
}


# ── Text splitting ────────────────────────────────────────────────

def split_of_sentences(text: str) -> list[str]:
    """Split text on sentence boundaries (.!?)."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s]


def chunk_text(text, chunk_size=800, overlap=100) -> list[str]:
    """Split long text into overlapping chunks, respecting paragraphs."""
    paragraph = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    def flush():
        nonlocal current
        if current:
            chunks.append(current)
            # keep the last `overlap` chars for context continuity
            current = current[-overlap:]

    for para in paragraph:
        if len(para) + len(current) + 1 <= chunk_size:
            current = f"{current}\n{para}".strip()
            continue

        if len(para) > chunk_size:
            for sentence in split_of_sentences(para):
                if len(current) + len(sentence) + 1 <= chunk_size:
                    current = f"{current} {sentence}".strip()
                else:
                    flush()
                    current = f"{current} {sentence}".strip()
        else:
            flush()
            current = f"{current}\n{para}".strip()

    if current:
        chunks.append(current)

    return chunks


# ── Public API ────────────────────────────────────────────────────

def ingest_file(file_path, chunk_size=800, overlap=100):
    """Read a single file and return a list of chunk dicts."""
    path = Path(file_path).resolve()

    reader = FORMAT_HANDLER.get(path.suffix.lower())

    if reader is None:
        print(f"Skipping unsupported file: {path.name}")
        return []

    try:
        text = reader(str(path))
    except Exception as e:
        print(f"Failed to read file {path.name}: {e}")
        return []

    if not text.strip():
        print(f"Not extractable text in: {path.name}")
        return []

    chunks = chunk_text(text, chunk_size, overlap)

    return [{
        "source": str(path),
        "chunk_index": i,
        "text": chunk_text_item,
    } for i, chunk_text_item in enumerate(chunks)]


def ingest_folder(folder_path, chunk_size=800, overlap=100):
    """Recursively ingest every supported file under *folder_path*."""
    path = Path(folder_path).resolve()
    all_chunks = []

    for file in path.glob("**/*"):
        if file.is_file():
            chunks = ingest_file(str(file), chunk_size, overlap)
            all_chunks.extend(chunks)

    return all_chunks
