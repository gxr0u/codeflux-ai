from __future__ import annotations

import os
from typing import List

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from app.config.debug import debug_print
from app.config.settings import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL


load_dotenv()

USE_OPENAI = OPENAI_API_KEY is not None and False
LOCAL_MODEL_NAME = "BAAI/bge-small-en-v1.5"
HF_TOKEN = (
    os.getenv("HF_TOKEN")
    or os.getenv("HUGGINGFACE_HUB_TOKEN")
    or os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

if USE_OPENAI:
    import openai

    openai.api_key = OPENAI_API_KEY


_local_model = None
_local_model_error = None


def _normalize_embeddings(vectors: List[List[float]]) -> List[List[float]]:
    embeddings = np.array(vectors, dtype="float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (embeddings / norms).tolist()


def _load_local_model(local_files_only: bool) -> SentenceTransformer:
    kwargs = {"local_files_only": local_files_only}
    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN
    return SentenceTransformer(LOCAL_MODEL_NAME, **kwargs)


def _get_local_model() -> SentenceTransformer:
    global _local_model, _local_model_error

    if _local_model is not None:
        return _local_model
    if _local_model_error is not None:
        raise RuntimeError(_local_model_error)

    try:
        _local_model = _load_local_model(local_files_only=True)
        return _local_model
    except Exception as cached_error:
        cached_message = str(cached_error)

    if HF_TOKEN:
        try:
            _local_model = _load_local_model(local_files_only=False)
            return _local_model
        except Exception as download_error:
            _local_model_error = (
                "Local embedding model is not cached and Hugging Face download failed. "
                f"Cache error: {cached_message}. Download error: {download_error}"
            )
            raise RuntimeError(_local_model_error) from download_error

    _local_model_error = (
        "Local embedding model is not cached and no Hugging Face token is configured. "
        "Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN to allow the initial download, "
        "or keep OPENAI_API_KEY configured so embeddings use OpenAI."
    )
    raise RuntimeError(_local_model_error)


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Prefer OpenAI embeddings when configured.
    Fall back to a cached local SentenceTransformer model.
    If a Hugging Face token is configured, allow one-time model download.
    """

    if USE_OPENAI:
        try:
            response = openai.embeddings.create(
                model=OPENAI_EMBEDDING_MODEL,
                input=texts,
            )
            return _normalize_embeddings([item.embedding for item in response.data])
        except Exception as exc:
            debug_print("[WARN] OpenAI embeddings failed, falling back to local model.")
            debug_print(f"[WARN] Reason: {exc}")

    model = _get_local_model()
    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype="float32").tolist()
