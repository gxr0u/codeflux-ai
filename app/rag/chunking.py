# app/rag/chunking.py

from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> List[str]:
    """
    Splits text into overlapping chunks.
    Keeps chunks small enough for embeddings while preserving context.
    """
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start = end - overlap

        if start < 0:
            start = 0

    return chunks
