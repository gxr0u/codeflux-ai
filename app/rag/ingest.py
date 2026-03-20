# app/rag/ingest.py

from pathlib import Path
import os
import pickle
import uuid
from typing import List, Dict
import re

import faiss
import numpy as np

from app.rag.chunking import chunk_text
from app.services.embedding_service import get_embeddings


# 🔹 Paths
BASE_DIR = Path(__file__).resolve().parents[2]
VECTOR_STORE_PATH = BASE_DIR / "data" / "vector_store"

INDEX_FILE = "index.faiss"
META_FILE = "metadata.pkl"


# =========================================================
# 🔹 LOAD DOCUMENTS (supports nested folders + .md)
# =========================================================
def load_documents(doc_dir: str) -> List[Dict]:
    documents = []

    for path in Path(doc_dir).rglob("*"):
        if path.suffix not in [".md", ".txt"]:
            continue

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        documents.append({
            "path": path,
            "text": text
        })

    return documents


# =========================================================
# 🔹 EXTRACT METADATA (CRITICAL FOR FILTERING)
# =========================================================
def extract_metadata(path: Path) -> Dict:
    parts = path.parts

    # Expected:
    # data/raw_docs/python/asyncio/sleep.md

    try:
        return {
            "language": parts[-3],
            "library": parts[-2],
            "name": path.stem,
            "source": str(path)
        }
    except IndexError:
        return {
            "language": "unknown",
            "library": "general",
            "name": path.stem,
            "source": str(path)
        }


# =========================================================
# 🔹 OPTIONAL: SIMPLE KEYWORD EXTRACTION
# =========================================================
def extract_keywords(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_.]*", text.lower())
    return list(set(tokens[:20]))

# =========================================================
# 🔹 INGEST PIPELINE
# =========================================================
def ingest_documents(doc_dir: str):
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

    print("🔄 Loading documents...")
    docs = load_documents(doc_dir)

    all_chunks = []
    metadata = []

    for doc in docs:
        chunks = chunk_text(doc["text"])
        doc_meta = extract_metadata(doc["path"])

        for chunk in chunks:
            chunk = "passage: " + chunk
            chunk_id = str(uuid.uuid4())

            all_chunks.append(chunk)

            metadata.append({
                "id": chunk_id,
                "text": chunk,
                **doc_meta,
                "has_code": "```" in chunk,
                "length": len(chunk),
                "keywords": extract_keywords(chunk)
            })

    print(f"[INGEST] Total chunks: {len(all_chunks)}")

    # =====================================================
    # 🔹 EMBEDDINGS
    # =====================================================
    print("🔄 Generating embeddings...")
    embeddings = get_embeddings(all_chunks)
    embeddings = np.array(embeddings).astype("float32")

    # 🔥 IMPORTANT: Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]

    # Use Inner Product (cosine similarity after normalization)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # =====================================================
    # 🔹 SAVE VECTOR STORE
    # =====================================================
    print("💾 Saving FAISS index...")
    faiss.write_index(index, str(VECTOR_STORE_PATH / INDEX_FILE))

    print("💾 Saving metadata...")
    with open(VECTOR_STORE_PATH / META_FILE, "wb") as f:
        pickle.dump(metadata, f)

    print("✅ Vector store created successfully!")


# =========================================================
# 🔹 MAIN
# =========================================================
if __name__ == "__main__":
    ingest_documents("data/raw_docs")