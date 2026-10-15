from pathlib import Path
from pypdf import PdfReader
from docx import Document
import os
import re


#reading the different docs based on the format
def read_pdf(document): 
    reader = PdfReader(document)
    return "\n".join((p.extract_text() or "") for p in reader.pages)

def read_docx(document):
    reader = Document(document)
    return '\n'.join([paragraph.text for paragraph in reader.paragraphs])

def read_txt(document):
    with open(document, encoding="utf8", errors="ignore") as f:
        return f.read()

def read_md(document): 
    return Path(document).read_text(encoding="utf8", errors="ignore")


FORMAT_HANDLER = {
    ".pdf": read_pdf, 
    ".docx": read_docx, 
    ".txt": read_txt, 
    ".md": read_md
}


#splitting the text and storing it in list for chunking 
def split_of_sentences(text: str) -> list[str]: 
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s]




def chunk_text(text, chunk_size = 800, overlap = 100) -> list[str]: 
    paragraph = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    def flush(): 
        nonlocal current
        if current: 
            chunks.append(current)
            current = current[-100:]

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


#main folder for chunking 
def ingest_file(file_path, chunk_size = 800, overlap = 100):
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
        print(f"Not extractable text in:{path.name}")
        return []

    chunks = chunk_text(text, chunk_size, overlap)

    return [{
        "source": str(path),
        "index": i,
        "text": text
    } for i, text in enumerate(chunks)]



def ingest_folder(folder_path, chunk_size = 800, overlap = 100):
    path = Path(folder_path).resolve()
    all_chunks = []

    for file in path.glob("**/*"):
        if file.is_file():
            chunks = ingest_file(str(file), chunk_size, overlap)
            all_chunks.extend(chunks)

    return all_chunks


