from __future__ import annotations

from functools import lru_cache
import re
from typing import Dict, Iterable, List, Optional, Tuple


LANGUAGE_PATTERNS = {
    "python": [
        r"\bpython\b",
        r"\bpy\b",
        r"\bpytest\b",
        r"\bpip\b",
        r"\basyncio\b",
        r"\bfastapi\b",
        r"\bdjango\b",
        r"\bflask\b",
        r"\bdef\s+\w+",
        r"\bfrom\s+\w+\s+import\b",
        r"\bself\b",
        r"\bprint\(",
        r"\bNone\b",
    ],
    "javascript": [
        r"\bjavascript\b",
        r"\bjs\b",
        r"\bnode(?:\.js)?\b",
        r"\bnpm\b",
        r"\bexpress\b",
        r"\breact\b",
        r"\bvue\b",
        r"\bnext\.?js\b",
        r"\bfunction\b",
        r"=>",
        r"\bconsole\.log\b",
        r"\brequire\(",
        r"\bmodule\.exports\b",
    ],
    "typescript": [
        r"\btypescript\b",
        r"\bts\b",
        r"\btsx\b",
        r"\binterface\b",
        r"\benum\b",
        r"\bimplements\b",
        r"\btype\s+\w+\s*=",
        r":\s*(string|number|boolean|void|unknown|never|any)\b",
        r"\bas\s+const\b",
        r"\breadonly\b",
    ],
}

LIBRARY_ALIASES = {
    "react": ["react", "reactjs", "jsx", "tsx", "useeffect", "usestate"],
    "next.js": ["next.js", "nextjs", "next/router", "app router"],
    "node": ["node", "nodejs", "node.js", "npm"],
    "express": ["express", "expressjs"],
    "asyncio": ["asyncio", "await", "async def"],
    "fastapi": ["fastapi"],
    "django": ["django"],
    "flask": ["flask"],
    "pytest": ["pytest"],
    "pandas": ["pandas", "dataframe", "series"],
    "numpy": ["numpy", "ndarray"],
    "vite": ["vite"],
    "webpack": ["webpack"],
    "tailwind": ["tailwind", "tailwindcss"],
}

INTENT_KEYWORDS = {
    "debugging": [
        "error",
        "exception",
        "traceback",
        "bug",
        "fix",
        "issue",
        "failing",
        "fails",
        "not working",
        "crash",
        "undefined",
    ],
    "implementation": [
        "implement",
        "build",
        "create",
        "write",
        "generate",
        "add",
        "how do i",
        "how to",
        "example",
        "sample",
    ],
    "concept": [
        "what is",
        "why",
        "explain",
        "difference",
        "compare",
        "when should",
        "concept",
    ],
    "optimization": [
        "optimize",
        "improve",
        "faster",
        "performance",
        "reduce latency",
        "speed up",
        "efficient",
    ],
    "refactor": [
        "refactor",
        "clean up",
        "restructure",
        "simplify",
    ],
}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "could",
    "do", "does", "for", "from", "get", "help", "i", "if", "in", "into", "is",
    "it", "me", "my", "of", "on", "or", "please", "should", "show", "so", "than",
    "that", "the", "this", "to", "use", "using", "want", "what", "when", "where",
    "which", "why", "with", "would", "you", "your", "how",
}

ERROR_PATTERNS = [
    r"(Traceback[\s\S]+?)(?=\n\s*\n|$)",
    r"((?:\w+\.)*\w*(?:Error|Exception)\s*:\s*[^\n]+)",
    r"((?:\w+\.)*\w*(?:Error|Exception))",
    r"((?:HTTP|Status)\s+\d{3}[^\n]*)",
    r"(`[^`]*(?:Error|Exception)[^`]*`)",
    r"(\"[^\"]*(?:error|exception|failed|undefined)[^\"]*\")",
    r"('[^']*(?:error|exception|failed|undefined)[^']*')",
]


def _normalize_text(*parts: Optional[str]) -> str:
    return "\n".join(part for part in parts if part).strip()


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _match_patterns(text: str, patterns: Iterable[str], weight: float = 1.0) -> float:
    score = 0.0
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            score += weight
    return score


def detect_language(query: str, code: str = None) -> Tuple[str, float]:
    text = _normalize_text(query, code).lower()
    scores = {
        language: _match_patterns(text, patterns)
        for language, patterns in LANGUAGE_PATTERNS.items()
    }

    # TypeScript should outrank JavaScript when TS-specific syntax exists.
    if scores["typescript"] > 0:
        scores["javascript"] = max(0.0, scores["javascript"] - 0.75)

    best_language = max(scores, key=scores.get)
    best_score = scores[best_language]
    second_score = max(score for lang, score in scores.items() if lang != best_language)

    if best_score == 0:
        return "unknown", 0.0

    confidence = 0.45 + min(best_score * 0.12, 0.4) + min((best_score - second_score) * 0.08, 0.15)
    return best_language, _clamp(confidence)


def detect_intent(query: str, error: Optional[str] = None) -> Tuple[str, float]:
    q = query.lower()
    scores = {
        intent: sum(1 for keyword in keywords if keyword in q)
        for intent, keywords in INTENT_KEYWORDS.items()
    }

    if error:
        scores["debugging"] += 2

    if re.search(r"\bhow\s+to\b|\bhow\s+do\s+i\b", q):
        scores["implementation"] += 1
    if re.search(r"\bvs\b|\bversus\b|\bdifference\b", q):
        scores["concept"] += 1

    best_intent = max(scores, key=scores.get)
    best_score = scores[best_intent]
    second_score = max(score for intent, score in scores.items() if intent != best_intent)

    if best_score == 0:
        return "general", 0.35

    confidence = 0.5 + min(best_score * 0.13, 0.3) + min((best_score - second_score) * 0.08, 0.15)
    return best_intent, _clamp(confidence)


def extract_libraries(query: str, code: str = None) -> List[str]:
    text = _normalize_text(query, code).lower()
    libraries = []
    for library, aliases in LIBRARY_ALIASES.items():
        if any(alias in text for alias in aliases):
            libraries.append(library)
    return sorted(set(libraries))


def extract_error(query: str, code: str = None) -> Optional[str]:
    text = _normalize_text(query, code)
    for pattern in ERROR_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue

        error = match.group(1).strip().strip("`\"'")
        error = re.sub(r"\s+", " ", error)
        error = re.sub(r"\s+in\s+(python|javascript|typescript|react|node)(?:\W.*)?$", "", error, flags=re.IGNORECASE)
        error = error.rstrip("?.!,;:")
        return error[:300]
    return None


def _tokenize_topic_candidates(text: str) -> List[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_./-]*", text)


def extract_topics(
    query: str,
    code: str = None,
    libraries: Optional[List[str]] = None,
    error: Optional[str] = None,
    limit: int = 8,
) -> List[str]:
    text = _normalize_text(query, code)
    lower_text = text.lower()
    libraries = libraries or []

    candidates: Dict[str, float] = {}

    # Score technical unigrams.
    for token in _tokenize_topic_candidates(text):
        normalized = token.strip("._-/").lower()
        if len(normalized) < 3 or normalized in STOPWORDS:
            continue
        if normalized in {"python", "javascript", "typescript"}:
            continue
        if normalized in libraries:
            continue

        score = 1.0
        if any(ch in token for ch in "._/-"):
            score += 0.8
        if "_" in token or re.search(r"[a-z][A-Z]", token):
            score += 0.8
        if token[0].isupper():
            score += 0.3
        candidates[normalized] = candidates.get(normalized, 0.0) + score

    # Score frequent bigrams and trigrams to get more semantic topics than raw tokens.
    words = [word.lower() for word in re.findall(r"\b[a-zA-Z][a-zA-Z0-9_]+\b", text)]
    filtered_words = [word for word in words if word not in STOPWORDS and len(word) > 2]
    for size, base_score in ((2, 2.4), (3, 2.0)):
        for index in range(len(filtered_words) - size + 1):
            phrase = " ".join(filtered_words[index:index + size])
            if any(part in {"error", "issue", "problem"} for part in phrase.split()):
                continue
            candidates[phrase] = candidates.get(phrase, 0.0) + base_score

    if error:
        for token in _tokenize_topic_candidates(error):
            normalized = token.lower()
            if len(normalized) > 2:
                candidates[normalized] = candidates.get(normalized, 0.0) + 1.5

    ranked = sorted(
        candidates.items(),
        key=lambda item: (
            -item[1],
            len(item[0].split()),
            lower_text.find(item[0]) if item[0] in lower_text else 10**6,
            item[0],
        ),
    )

    topics: List[str] = []
    for topic, _ in ranked:
        if topic in topics:
            continue
        topics.append(topic)
        if len(topics) >= limit:
            break

    return topics


def _infer_requires_retrieval(
    query: str,
    intent: str,
    libraries: List[str],
    error: Optional[str],
    needs_code: bool,
) -> bool:
    q = query.lower()

    doc_signals = [
        "documentation",
        "docs",
        "api",
        "reference",
        "version",
        "breaking change",
        "configuration",
        "best practice",
        "compare",
        "difference between",
    ]

    no_retrieval_signals = [
        "write code",
        "generate code",
        "fix this code",
        "debug this",
        "explain this code",
    ]

    if any(signal in q for signal in no_retrieval_signals):
        return False
    if any(signal in q for signal in doc_signals):
        return True
    if intent == "concept" and libraries:
        return True
    if intent == "debugging" and (error or needs_code):
        return False
    if intent in {"implementation", "optimization"} and libraries:
        return True
    return len(query.split()) >= 14 and not needs_code


@lru_cache(maxsize=256)
def _analyze_query_cached(query: str, code: str = "") -> Dict:
    error = extract_error(query, code)
    libraries = extract_libraries(query, code)
    language, language_confidence = detect_language(query, code)
    intent, intent_confidence = detect_intent(query, error=error)
    topics = extract_topics(query, code, libraries=libraries, error=error)
    needs_code = code is not None or bool(re.search(r"\b(code|snippet|function|class|file)\b", query.lower()))
    requires_retrieval = _infer_requires_retrieval(query, intent, libraries, error, needs_code)

    analysis = {
        "intent": intent,
        "language": language,
        "topics": topics,
        "libraries": libraries,
        "error": error,
        "needs_code": needs_code,
        "requires_retrieval": requires_retrieval,
        "confidence": {
            "intent": intent_confidence,
            "language": language_confidence,
        },
    }

    return analysis


def analyze_query(query: str, code: str = None) -> Dict:
    cached = _analyze_query_cached(query, code or "")
    return {
        **cached,
        "topics": list(cached["topics"]),
        "libraries": list(cached["libraries"]),
        "confidence": dict(cached["confidence"]),
    }
