from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "i",
    "if", "in", "into", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "what", "when", "where", "why", "with",
}

FIX_HINTS = (
    "fix",
    "solution",
    "workaround",
    "resolve",
    "resolved",
    "avoid",
    "set ",
    "use ",
    "pass ",
    "ensure",
    "check ",
    "install",
    "upgrade",
    "downgrade",
    "configure",
    "return ",
)

EXAMPLE_HINTS = (
    "example",
    "for example",
    "e.g.",
    "snippet",
    "sample",
    "usage",
    "```",
)

API_PATTERNS = (
    r"\b[A-Za-z_][A-Za-z0-9_]*\(",
    r"\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\b",
    r"\b(?:GET|POST|PUT|PATCH|DELETE)\s+/[A-Za-z0-9_./:-]*",
    r"\b/[A-Za-z0-9_./:-]+\b",
)


def _tokenize(text: str) -> List[str]:
    return [
        token for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_./-]*", text.lower())
        if token not in STOPWORDS
    ]


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_bullet(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^[-*]\s*", "", text)
    return _normalize_whitespace(text)


def _dedupe_items(items: Sequence[str], limit: int) -> List[str]:
    deduped: List[str] = []
    seen_signatures = set()

    for item in items:
        normalized = _normalize_bullet(item)
        if not normalized:
            continue

        signature_tokens = sorted(set(_tokenize(normalized)))
        signature = " ".join(signature_tokens[:12]) or normalized.lower()
        if signature in seen_signatures:
            continue

        if any(
            normalized.lower() in existing.lower() or existing.lower() in normalized.lower()
            for existing in deduped
        ):
            continue

        seen_signatures.add(signature)
        deduped.append(f"- {normalized}")
        if len(deduped) >= limit:
            break

    return deduped


def _extract_code_blocks(text: str) -> List[str]:
    blocks = re.findall(r"```[\s\S]*?```", text)
    return [block.strip() for block in blocks]


def _extract_sentences(text: str) -> List[str]:
    protected = text.replace("```", "\n```")
    parts = re.split(r"(?<=[.!?])\s+|\n+", protected)
    return [part.strip() for part in parts if part.strip()]


def _query_terms(query: str) -> set[str]:
    return set(_tokenize(query))


def _extract_function_and_api_refs(text: str) -> List[str]:
    refs = []
    for pattern in API_PATTERNS:
        refs.extend(re.findall(pattern, text))
    cleaned = []
    for ref in refs:
        ref = ref.rstrip("(")
        if len(ref) >= 3:
            cleaned.append(ref)
    return cleaned


def _sentence_score(query_terms: set[str], sentence: str) -> float:
    lowered = sentence.lower()
    tokens = set(_tokenize(sentence))
    overlap = len(tokens & query_terms)

    score = overlap * 2.0
    if any(hint in lowered for hint in FIX_HINTS):
        score += 2.0
    if any(hint in lowered for hint in EXAMPLE_HINTS):
        score += 1.5
    if re.search(r"`[^`]+`", sentence):
        score += 1.2
    if any(re.search(pattern, sentence) for pattern in API_PATTERNS):
        score += 1.4
    if len(sentence) > 240:
        score -= 0.8
    return score


def _classify_sentence(sentence: str) -> str:
    lowered = sentence.lower()
    if any(hint in lowered for hint in FIX_HINTS):
        return "fixes"
    if any(hint in lowered for hint in EXAMPLE_HINTS) or "```" in sentence:
        return "examples"
    if any(re.search(pattern, sentence) for pattern in API_PATTERNS):
        return "concepts"
    return "concepts"


def _trim_sentence(sentence: str, limit: int = 180) -> str:
    sentence = _normalize_whitespace(sentence)
    if len(sentence) <= limit:
        return sentence
    return sentence[: limit - 3].rstrip() + "..."


def _format_code_example(block: str, max_lines: int = 8) -> str:
    lines = block.strip().splitlines()
    if len(lines) <= max_lines:
        return block.strip()

    trimmed = lines[:max_lines]
    if trimmed[-1].strip() != "```":
        trimmed.append("...")
        if lines[-1].strip() == "```":
            trimmed.append("```")
    return "\n".join(trimmed)


def compress_docs(query: str, docs: List[Dict], llm=None) -> Dict[str, List[str]]:
    """
    Deterministic extractive compressor optimized for small LLM contexts.

    `llm` is kept for backward compatibility and intentionally unused here to
    avoid extra latency and hallucination risk.
    """
    del llm

    query_terms = _query_terms(query)
    concepts: List[Tuple[float, str]] = []
    examples: List[Tuple[float, str]] = []
    fixes: List[Tuple[float, str]] = []

    for doc in docs:
        text = doc.get("text", "")
        if not text:
            continue

        refs = _extract_function_and_api_refs(text)
        for ref in refs[:8]:
            concepts.append((1.8, f"API/function reference: `{ref}`"))

        for block in _extract_code_blocks(text):
            block_score = 2.5 + len(query_terms & set(_tokenize(block)))
            examples.append((block_score, _format_code_example(block)))

        for sentence in _extract_sentences(text):
            score = _sentence_score(query_terms, sentence)
            if score <= 0:
                continue

            sentence_type = _classify_sentence(sentence)
            trimmed = _trim_sentence(sentence)
            if sentence_type == "fixes":
                fixes.append((score, trimmed))
            elif sentence_type == "examples":
                examples.append((score, trimmed))
            else:
                concepts.append((score, trimmed))

    concepts_sorted = [item for _, item in sorted(concepts, key=lambda entry: entry[0], reverse=True)]
    examples_sorted = [item for _, item in sorted(examples, key=lambda entry: entry[0], reverse=True)]
    fixes_sorted = [item for _, item in sorted(fixes, key=lambda entry: entry[0], reverse=True)]

    return {
        "concepts": _dedupe_items(concepts_sorted, limit=6),
        "examples": _dedupe_items(examples_sorted, limit=4),
        "fixes": _dedupe_items(fixes_sorted, limit=6),
    }
