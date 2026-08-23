"""
Document → text → clean → chunk utilities for the RAG knowledge base.
Supports PDF (via pypdf) and TXT. Add more formats here if you have time
left, following the same extract_text(path) -> str pattern.
"""
import os
import re


def extract_text(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    elif ext == ".txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {ext} (only .pdf and .txt are supported)")


def clean_text(text):
    """Collapse whitespace/newlines from PDF extraction artifacts."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text, chunk_words=150, overlap_words=30):
    """Simple fixed-size word chunking with overlap. Good enough for a
    prototype knowledge base; a real production system would chunk on
    sentence/section boundaries instead."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    step = max(1, chunk_words - overlap_words)
    while start < len(words):
        end = start + chunk_words
        chunk = " ".join(words[start:end])
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks
