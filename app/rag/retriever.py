from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import json
import math
import os
import pickle
import re
from typing import Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np

from app.config.debug import debug_print
from app.rag.reranker import RerankWeights, rerank

try:
    from rank_bm25 import BM25Okapi
except Exception:
    class BM25Okapi:  # type: ignore[no-redef]
        def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
            self.corpus = corpus
            self.k1 = k1
            self.b = b
            self.doc_len = [len(doc) for doc in corpus]
            self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
            self.doc_freqs: List[Dict[str, int]] = []
            self.idf: Dict[str, float] = {}

            nd: Dict[str, int] = {}
            for doc in corpus:
                frequencies: Dict[str, int] = {}
                for word in doc:
                    frequencies[word] = frequencies.get(word, 0) + 1
                self.doc_freqs.append(frequencies)
                for word in frequencies:
                    nd[word] = nd.get(word, 0) + 1

            corpus_size = len(corpus)
            for word, freq in nd.items():
                self.idf[word] = math.log(1 + (corpus_size - freq + 0.5) / (freq + 0.5))

        def get_scores(self, query_tokens: List[str]) -> np.ndarray:
            scores = np.zeros(len(self.corpus), dtype="float32")
            for index, frequencies in enumerate(self.doc_freqs):
                doc_length = self.doc_len[index] or 1
                for token in query_tokens:
                    if token not in frequencies:
                        continue
                    idf = self.idf.get(token, 0.0)
                    freq = frequencies[token]
                    denominator = freq + self.k1 * (1 - self.b + self.b * doc_length / max(self.avgdl, 1))
                    scores[index] += idf * ((freq * (self.k1 + 1)) / denominator)
            return scores


BASE_DIR = Path(__file__).resolve().parents[2]
VECTOR_STORE_PATH = BASE_DIR / "data" / "vector_store"

INDEX_FILE = "index.faiss"
META_FILE = "metadata.pkl"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "code", "debug", "for",
    "from", "how", "i", "in", "is", "it", "my", "of", "on", "or", "slow",
    "the", "this", "to", "what", "why", "with",
}


def _get_embeddings(texts: Sequence[str]) -> List[List[float]]:
    from app.services.embedding_service import get_embeddings

    return get_embeddings(list(texts))


@lru_cache(maxsize=128)
def _cached_embedding(query: str) -> np.ndarray:
    emb = _get_embeddings([query])[0]
    return np.array(emb, dtype="float32")

def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_./-]*", text.lower())

    expanded = []
    for token in tokens:
        expanded.append(token)

        # split dotted tokens
        if "." in token:
            expanded.extend(token.split("."))

    return [t for t in expanded if t not in STOPWORDS]


def _normalize_doc(document: Dict) -> Dict:
    normalized = dict(document)
    metadata = normalized.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    for key in ("source", "language", "library", "name", "id", "has_code"):
        if key in normalized and key not in metadata:
            metadata[key] = normalized[key]
    normalized["metadata"] = metadata
    normalized["score"] = float(normalized.get("score", 0.0))
    return normalized


def _metadata_value(document: Dict, key: str):
    metadata = document.get("metadata") or {}
    if isinstance(metadata, dict) and key in metadata:
        return metadata[key]
    return document.get(key)


def _combined_doc_text(document: Dict) -> str:
    metadata = document.get("metadata") or {}
    parts = [document.get("text", ""), document.get("source", "")]
    if isinstance(metadata, dict):
        parts.extend(str(value) for value in metadata.values())
    return " ".join(part for part in parts if part).lower()


def _doc_identity(document: Dict) -> str:
    for key in ("id", "doc_id", "chunk_id", "source", "name"):
        value = _metadata_value(document, key)
        if value:
            return f"{key}:{value}"
    text = re.sub(r"\s+", " ", document.get("text", "").strip().lower())
    return f"text:{text[:180]}"


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
    return ((values - minimum) / (maximum - minimum)).astype("float32").tolist()


def initialize_bm25(metadata: List[Dict]):
    corpus = [doc.get("text", "") for doc in metadata]
    tokenized = [_tokenize(text) for text in corpus]
    return BM25Okapi(tokenized)


def bm25_search(query: str, metadata: List[Dict], bm25, top_k: int = 10) -> List[Dict]:
    query_tokens = _tokenize(query)
    if not query_tokens or not metadata:
        return []

    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for index in top_indices:
        score = float(scores[index])
        if score <= 0:
            continue
        item = _normalize_doc(metadata[int(index)])
        item["score"] = score
        results.append(item)
    return results


def vector_search(
    query_embedding: np.ndarray,
    index,
    metadata: List[Dict],
    top_k: int = 10,
) -> List[Dict]:
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    query_embedding = query_embedding.astype("float32")
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        item = _normalize_doc(metadata[int(idx)])
        item["score"] = float(score)
        results.append(item)
    return results


def apply_filters(docs: List[Dict], filters: Dict = None) -> List[Dict]:
    if not filters:
        return [_normalize_doc(doc) for doc in docs]

    normalized_filters = {}
    for key, value in filters.items():
        normalized_key = "library" if key == "libraries" else key
        normalized_filters[normalized_key] = value

    filtered = []
    for doc in docs:
        normalized = _normalize_doc(doc)
        keep = True
        for key, expected in normalized_filters.items():
            actual = _metadata_value(normalized, key)
            if actual is None:
                keep = False
                break

            actual_normalized = str(actual).strip().lower()
            if isinstance(expected, (list, tuple, set)):
                expected_values = {str(item).strip().lower() for item in expected}
                if actual_normalized not in expected_values:
                    keep = False
                    break
            else:
                if actual_normalized != str(expected).strip().lower():
                    keep = False
                    break

        if keep:
            filtered.append(normalized)

    return filtered


def _merge_hybrid_results(
    dense_docs: List[Dict],
    sparse_docs: List[Dict],
    query: str,
    dense_weight: float,
    sparse_weight: float,
) -> List[Dict]:
    dense_norm = _normalize_scores([doc["score"] for doc in dense_docs])
    sparse_norm = _normalize_scores([doc["score"] for doc in sparse_docs])

    merged: Dict[str, Dict] = {}
    query_lower = query.lower()

    for doc, score in zip(dense_docs, dense_norm):
        doc_id = _doc_identity(doc)
        merged[doc_id] = {**doc, "score": dense_weight * score}

    for doc, score in zip(sparse_docs, sparse_norm):
        doc_id = _doc_identity(doc)
        contribution = sparse_weight * score
        if doc_id in merged:
            merged[doc_id]["score"] += contribution
        else:
            merged[doc_id] = {**doc, "score": contribution}

    # Lightweight metadata-aware boosts.
    for doc in merged.values():
        library = str(_metadata_value(doc, "library") or "").lower()
        if library and library in query_lower:
            doc["score"] += 0.05
            
            # 🔥 ADD THIS (soft language boost)
        language = str(_metadata_value(doc, "language") or "").lower()
        if language and language in query_lower:
            doc["score"] += 0.08

        has_code = _metadata_value(doc, "has_code")
        if has_code and any(token in query_lower for token in ("debug", "error", "fix", "traceback")):
            doc["score"] += 0.03

    results = list(merged.values())
    results.sort(key=lambda item: item["score"], reverse=True)
    return results


def hybrid_retrieve(
    query: str,
    query_embedding: np.ndarray,
    index,
    metadata: List[Dict],
    bm25,
    filters: Dict = None,
    top_k: int = 10,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
) -> List[Dict]:
    dense_docs = vector_search(query_embedding, index, metadata, top_k=max(top_k * 4, 20))
    sparse_docs = bm25_search(query, metadata, bm25, top_k=max(top_k * 4, 20))

    dense_docs = apply_filters(dense_docs, filters)
    sparse_docs = apply_filters(sparse_docs, filters)

    merged = _merge_hybrid_results(
        dense_docs=dense_docs,
        sparse_docs=sparse_docs,
        query=query,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
    )
    return merged[:top_k]


def _text_similarity(left: str, right: str) -> float:
    left_tokens = set(_tokenize(left))
    right_tokens = set(_tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def deduplicate_docs(docs: List[Dict]) -> List[Dict]:
    kept: List[Dict] = []
    identity_map: Dict[str, int] = {}

    for doc in sorted((_normalize_doc(doc) for doc in docs), key=lambda item: item["score"], reverse=True):
        identity = _doc_identity(doc)
        existing_index = identity_map.get(identity)
        if existing_index is not None:
            if doc["score"] > kept[existing_index]["score"]:
                kept[existing_index] = doc
            continue

        duplicate_index = None
        for index, existing in enumerate(kept):
            if _text_similarity(doc.get("text", ""), existing.get("text", "")) >= 0.92:
                duplicate_index = index
                break

        if duplicate_index is not None:
            if doc["score"] > kept[duplicate_index]["score"]:
                kept[duplicate_index] = doc
            continue

        identity_map[identity] = len(kept)
        kept.append(doc)

    kept.sort(key=lambda item: item["score"], reverse=True)
    return kept


def reciprocal_rank_fusion(doc_lists: List[List[Dict]], k: int = 60) -> List[Dict]:
    fused_scores: Dict[str, float] = {}
    best_docs: Dict[str, Dict] = {}

    for doc_list in doc_lists:
        for rank, doc in enumerate(doc_list, start=1):
            normalized = _normalize_doc(doc)
            doc_id = _doc_identity(normalized)
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

            current = best_docs.get(doc_id)
            if current is None or normalized["score"] > current["score"]:
                best_docs[doc_id] = normalized

    fused_results = []
    for doc_id, doc in best_docs.items():
        fused_doc = dict(doc)
        fused_doc["score"] = fused_scores[doc_id]
        fused_results.append(fused_doc)

    fused_results.sort(key=lambda item: item["score"], reverse=True)
    return fused_results


def _simple_terms(query: str) -> List[str]:
    terms = []
    for token in _tokenize(query):
        if token not in terms:
            terms.append(token)
    return terms[:4]


@lru_cache(maxsize=256)
def _generate_query_variants_cached(query: str, analysis_key: str) -> Tuple[str, ...]:
    analysis = json.loads(analysis_key) if analysis_key else {}
    language = analysis.get("language")
    libraries = analysis.get("libraries", [])[:2]
    topics = analysis.get("topics", [])[:2]
    intent = analysis.get("intent", "general")
    error = analysis.get("error")

    variants: List[str] = [query.strip()]

    simplified_parts = _simple_terms(query)
    if language and language != "unknown":
        simplified_parts.append(language)
    simplified_parts.extend(libraries)
    simplified = " ".join(dict.fromkeys(part for part in simplified_parts if part))
    if simplified:
        variants.append(simplified)

    expanded_parts = [query.strip()]
    if language and language != "unknown":
        expanded_parts.append(language)
    expanded_parts.extend(libraries)
    expanded_parts.extend(topics)
    if error:
        expanded_parts.append(error)
    expanded = " ".join(dict.fromkeys(part for part in expanded_parts if part))
    variants.append(expanded)

    if intent == "debugging":
        intent_parts = ["how to fix", *(libraries or []), *(topics or []), *( [error] if error else [] )]
        intent_variant = " ".join(part for part in intent_parts if part)
        if intent_variant.strip():
            variants.append(intent_variant.strip())
    elif intent == "optimization":
        intent_parts = ["performance issue", *(libraries or []), *(topics or []), language or ""]
        intent_variant = " ".join(part for part in intent_parts if part)
        if intent_variant.strip():
            variants.append(intent_variant.strip())
    elif intent == "implementation":
        intent_parts = ["how to implement", *(libraries or []), *(topics or []), language or ""]
        intent_variant = " ".join(part for part in intent_parts if part)
        if intent_variant.strip():
            variants.append(intent_variant.strip())
    elif intent == "concept":
        intent_parts = ["explain", *(libraries or []), *(topics or []), language or ""]
        intent_variant = " ".join(part for part in intent_parts if part)
        if intent_variant.strip():
            variants.append(intent_variant.strip())

    deduped: List[str] = []
    seen = set()
    for variant in variants:
        normalized = re.sub(r"\s+", " ", variant).strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
        if len(deduped) >= 5:
            break

    return tuple(deduped[:5])


def generate_query_variants(query: str, analysis: dict) -> List[str]:
    analysis_subset = {
        "intent": analysis.get("intent"),
        "language": analysis.get("language"),
        "libraries": analysis.get("libraries", []),
        "topics": analysis.get("topics", []),
        "error": analysis.get("error"),
    }
    analysis_key = json.dumps(analysis_subset, sort_keys=True)
    variants = list(_generate_query_variants_cached(query, analysis_key))
    return variants[: max(3, min(5, len(variants)))]


def _code_signal_boost(document: Dict, code_info: Optional[Dict]) -> float:
    if not code_info:
        return 0.0

    haystack = _combined_doc_text(document)
    boost = 0.0

    for error in code_info.get("errors", []):
        if error.lower() in haystack:
            boost += 0.16
    for library in code_info.get("libraries", []):
        if library.lower() in haystack:
            boost += 0.09
    for function_name in code_info.get("functions", []):
        if function_name.lower() in haystack:
            boost += 0.1
    for pattern in code_info.get("patterns", []):
        normalized = pattern.lower().replace("_", " ")
        if normalized in haystack or pattern.lower() in haystack:
            boost += 0.05

    return min(boost, 0.35)


class Retriever:
    def __init__(self):
        index_path = os.path.join(VECTOR_STORE_PATH, INDEX_FILE)
        meta_path = os.path.join(VECTOR_STORE_PATH, META_FILE)

        if not os.path.exists(index_path):
            raise RuntimeError("FAISS index not found. Run ingestion first.")

        self.index = faiss.read_index(index_path)

        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        if len(self.metadata) < 20:
            debug_print("[WARN] Very small knowledge base — retrieval quality may be poor")

        self.bm25 = initialize_bm25(self.metadata)

    def hybrid_retrieve(
        self,
        query: str,
        top_k: int = 8,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        query = "query: " + query
        query_embedding = _cached_embedding(query)
        docs = hybrid_retrieve(
            query=query,
            query_embedding=query_embedding,
            index=self.index,
            metadata=self.metadata,
            bm25=self.bm25,
            filters=filters,
            top_k=top_k,
        )
        return [_normalize_doc(doc) for doc in docs]

    def multi_query_retrieve(
        self,
        query: str,
        analysis: Dict,
        filters: Optional[Dict] = None,
        top_k: int = 4,
    ) -> List[Dict]:
        queries = generate_query_variants(query, analysis)
        queries = ["query: " + q for q in queries]

        query_embeddings = np.array([_cached_embedding(q) for q in queries], dtype="float32")
        faiss.normalize_L2(query_embeddings)

        doc_lists: List[List[Dict]] = []
        for variant, embedding in zip(queries, query_embeddings):
            doc_lists.append(
                hybrid_retrieve(
                    query=variant,
                    query_embedding=embedding,
                    index=self.index,
                    metadata=self.metadata,
                    bm25=self.bm25,
                    filters=filters,
                    top_k=max(top_k * 2, 6),
                )
            )

        fused = reciprocal_rank_fusion(doc_lists)
        return deduplicate_docs(fused)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        code_info: Optional[Dict] = None,
        filters: Optional[Dict] = None,
        analysis: Optional[Dict] = None,
    ) -> List[Dict]:
        if analysis:
            docs = self.multi_query_retrieve(query, analysis=analysis, filters=filters, top_k=top_k)
        else:
            docs = self.hybrid_retrieve(query, top_k=top_k, filters=filters)

        docs = rerank(query, docs, top_k=top_k, weights=RerankWeights(keyword=0.35, embedding=0.65))

        results = []
        for doc in docs:
            normalized = _normalize_doc(doc)
            normalized["score"] = normalized["score"] * (1 + _code_signal_boost(normalized, code_info))
            normalized["source"] = _metadata_value(normalized, "source")
            results.append(normalized)

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]
