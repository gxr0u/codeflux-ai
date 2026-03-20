from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_CROSS_ENCODER_MODEL = "BAAI/bge-reranker-base"
_cross_encoder_cache: Dict[str, object] = {}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "i",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "what",
    "when", "why", "with",
}


@dataclass(frozen=True)
class RerankWeights:
    keyword: float = 0.35
    embedding: float = 0.65
    cross_encoder: float = 0.0


def _tokenize(text: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_./-]*", text.lower())
        if token not in STOPWORDS
    ]


def keyword_overlap_score(query: str, doc: str) -> float:
    query_tokens = _tokenize(query)
    doc_tokens = _tokenize(doc)
    if not query_tokens or not doc_tokens:
        return 0.0

    query_counts: Dict[str, int] = {}
    doc_counts: Dict[str, int] = {}
    for token in query_tokens:
        query_counts[token] = query_counts.get(token, 0) + 1
    for token in doc_tokens:
        doc_counts[token] = doc_counts.get(token, 0) + 1

    overlap = sum(min(query_counts[token], doc_counts.get(token, 0)) for token in query_counts)
    coverage = overlap / len(query_tokens)

    query_set = set(query_tokens)
    doc_set = set(doc_tokens)
    union = len(query_set | doc_set) or 1
    jaccard = len(query_set & doc_set) / union
    return (0.7 * coverage) + (0.3 * jaccard)


def _normalize_scores(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []

    values = np.array(scores, dtype="float32")
    minimum = float(values.min())
    maximum = float(values.max())
    if math.isclose(minimum, maximum):
        if maximum <= 0:
            return [0.0 for _ in scores]
        return [1.0 for _ in scores]

    normalized = (values - minimum) / (maximum - minimum)
    return normalized.astype("float32").tolist()


def _normalize_weights(weights: RerankWeights) -> RerankWeights:
    total = weights.keyword + weights.embedding + weights.cross_encoder
    if total <= 0:
        return RerankWeights()

    return RerankWeights(
        keyword=weights.keyword / total,
        embedding=weights.embedding / total,
        cross_encoder=weights.cross_encoder / total,
    )


def _cosine_similarity(query_embedding: np.ndarray, document_embeddings: np.ndarray) -> List[float]:
    if document_embeddings.size == 0:
        return []

    query_norm = np.linalg.norm(query_embedding)
    doc_norms = np.linalg.norm(document_embeddings, axis=1)
    denominator = np.maximum(query_norm * doc_norms, 1e-12)
    similarities = document_embeddings @ query_embedding / denominator
    return similarities.astype("float32").tolist()


def _resolve_top_k(top_k: Optional[int], total_documents: int) -> int:
    if total_documents <= 0:
        return 0
    if top_k is None:
        return total_documents
    return max(1, min(top_k, total_documents))


def _prepare_results(documents: Sequence[Dict], scores: Sequence[float]) -> List[Dict]:
    results = []
    for document, score in zip(documents, scores):
        metadata = document.get("metadata")
        if metadata is None:
            source = document.get("source")
            metadata = {"source": source} if source is not None else {}
        results.append({
            "text": document.get("text", ""),
            "metadata": metadata,
            "score": float(score),
        })
    return results


def _load_cross_encoder(model_name: str = DEFAULT_CROSS_ENCODER_MODEL):
    if model_name in _cross_encoder_cache:
        return _cross_encoder_cache[model_name]

    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name)
    _cross_encoder_cache[model_name] = model
    return model


def _get_embeddings(texts: Sequence[str]) -> List[List[float]]:
    from app.services.embedding_service import get_embeddings

    return get_embeddings(list(texts))


def _cross_encoder_scores(
    query: str,
    documents: Sequence[Dict],
    model_name: str,
    batch_size: int,
) -> List[float]:
    if not documents:
        return []

    model = _load_cross_encoder(model_name)
    pairs = [(query, document.get("text", "")) for document in documents]
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    return np.asarray(scores, dtype="float32").tolist()


def rerank(
    query: str,
    documents: List[Dict],
    top_k: Optional[int] = 5,
    *,
    weights: Optional[RerankWeights] = None,
    use_cross_encoder: bool = False,
    cross_encoder_model: str = DEFAULT_CROSS_ENCODER_MODEL,
    cross_encoder_batch_size: int = 16,
    cross_encoder_top_n: Optional[int] = None,
) -> List[Dict]:
    """
    Rerank retrieved documents with normalized lexical and semantic scores.

    Output:
    [
      {
        "text": "...",
        "metadata": {...},
        "score": float
      }
    ]
    """
    if not documents:
        return []

    weights = _normalize_weights(
        weights or RerankWeights(cross_encoder=0.2 if use_cross_encoder else 0.0)
    )
    rerank_top_k = _resolve_top_k(top_k, len(documents))

    document_texts = [document.get("text", "") for document in documents]
    keyword_scores = [keyword_overlap_score(query, text) for text in document_texts]

    embedding_vectors = np.asarray(_get_embeddings([query, *document_texts]), dtype="float32")
    query_embedding = embedding_vectors[0]
    document_embeddings = embedding_vectors[1:]
    semantic_scores = _cosine_similarity(query_embedding, document_embeddings)

    normalized_keyword_scores = _normalize_scores(keyword_scores)
    normalized_semantic_scores = _normalize_scores(semantic_scores)
    final_scores = [
        (weights.keyword * keyword_score) + (weights.embedding * semantic_score)
        for keyword_score, semantic_score in zip(normalized_keyword_scores, normalized_semantic_scores)
    ]

    ranked_indices = sorted(
        range(len(documents)),
        key=lambda index: final_scores[index],
        reverse=True,
    )

    if use_cross_encoder and weights.cross_encoder > 0:
        shortlist_size = _resolve_top_k(
            cross_encoder_top_n or max(rerank_top_k * 3, min(20, len(documents))),
            len(documents),
        )
        shortlist_indices = ranked_indices[:shortlist_size]
        shortlist_docs = [documents[index] for index in shortlist_indices]
        cross_scores = _cross_encoder_scores(
            query,
            shortlist_docs,
            model_name=cross_encoder_model,
            batch_size=cross_encoder_batch_size,
        )
        normalized_cross_scores = _normalize_scores(cross_scores)

        non_cross_weight = weights.keyword + weights.embedding
        for local_index, document_index in enumerate(shortlist_indices):
            base_score = final_scores[document_index]
            if non_cross_weight > 0:
                base_score = base_score / non_cross_weight
            final_scores[document_index] = (
                (weights.keyword + weights.embedding) * base_score
                + (weights.cross_encoder * normalized_cross_scores[local_index])
            )

        ranked_indices = sorted(
            range(len(documents)),
            key=lambda index: final_scores[index],
            reverse=True,
        )

    top_indices = ranked_indices[:rerank_top_k]
    top_documents = [documents[index] for index in top_indices]
    top_scores = [final_scores[index] for index in top_indices]
    return _prepare_results(top_documents, top_scores)


def rerank_batch(
    queries: Sequence[str],
    document_batches: Sequence[Sequence[Dict]],
    top_k: Optional[int] = 5,
    *,
    weights: Optional[RerankWeights] = None,
    use_cross_encoder: bool = False,
    cross_encoder_model: str = DEFAULT_CROSS_ENCODER_MODEL,
    cross_encoder_batch_size: int = 16,
    cross_encoder_top_n: Optional[int] = None,
) -> List[List[Dict]]:
    if len(queries) != len(document_batches):
        raise ValueError("queries and document_batches must have the same length")

    if not queries:
        return []

    weights = _normalize_weights(
        weights or RerankWeights(cross_encoder=0.2 if use_cross_encoder else 0.0)
    )

    flat_texts: List[str] = []
    offsets: List[Tuple[int, int]] = []
    for batch in document_batches:
        start = len(flat_texts)
        texts = [document.get("text", "") for document in batch]
        flat_texts.extend(texts)
        offsets.append((start, len(texts)))

    embeddings = np.asarray(_get_embeddings([*queries, *flat_texts]), dtype="float32")
    query_embeddings = embeddings[:len(queries)]
    document_embeddings = embeddings[len(queries):]

    results: List[List[Dict]] = []
    for batch_index, query in enumerate(queries):
        documents = list(document_batches[batch_index])
        if not documents:
            results.append([])
            continue

        start, count = offsets[batch_index]
        batch_document_embeddings = document_embeddings[start:start + count]
        keyword_scores = [
            keyword_overlap_score(query, document.get("text", ""))
            for document in documents
        ]
        semantic_scores = _cosine_similarity(query_embeddings[batch_index], batch_document_embeddings)
        normalized_keyword_scores = _normalize_scores(keyword_scores)
        normalized_semantic_scores = _normalize_scores(semantic_scores)
        final_scores = [
            (weights.keyword * keyword_score) + (weights.embedding * semantic_score)
            for keyword_score, semantic_score in zip(normalized_keyword_scores, normalized_semantic_scores)
        ]

        ranked_indices = sorted(
            range(len(documents)),
            key=lambda index: final_scores[index],
            reverse=True,
        )

        if use_cross_encoder and weights.cross_encoder > 0:
            rerank_top_k = _resolve_top_k(top_k, len(documents))
            shortlist_size = _resolve_top_k(
                cross_encoder_top_n or max(rerank_top_k * 3, min(20, len(documents))),
                len(documents),
            )
            shortlist_indices = ranked_indices[:shortlist_size]
            shortlist_docs = [documents[index] for index in shortlist_indices]
            cross_scores = _cross_encoder_scores(
                query,
                shortlist_docs,
                model_name=cross_encoder_model,
                batch_size=cross_encoder_batch_size,
            )
            normalized_cross_scores = _normalize_scores(cross_scores)

            non_cross_weight = weights.keyword + weights.embedding
            for local_index, document_index in enumerate(shortlist_indices):
                base_score = final_scores[document_index]
                if non_cross_weight > 0:
                    base_score = base_score / non_cross_weight
                final_scores[document_index] = (
                    (weights.keyword + weights.embedding) * base_score
                    + (weights.cross_encoder * normalized_cross_scores[local_index])
                )

            ranked_indices = sorted(
                range(len(documents)),
                key=lambda index: final_scores[index],
                reverse=True,
            )

        resolved_top_k = _resolve_top_k(top_k, len(documents))
        top_indices = ranked_indices[:resolved_top_k]
        results.append(
            _prepare_results(
                [documents[index] for index in top_indices],
                [final_scores[index] for index in top_indices],
            )
        )

    return results
