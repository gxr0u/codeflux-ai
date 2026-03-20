from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from app.agent.code_context import analyze_code_context
from app.agent.pipeline import decide_pipeline
from app.agent.prompt_builder import build_prompt
from app.agent.query_analyzer import analyze_query
from app.agent.tools import retriever
from app.agent.validation import post_process_response, refine_response, validate_response
from app.config.debug import debug_print
from app.config.settings import FALLBACK_TOP_K, SCORE_THRESHOLD
from app.rag.compressor import compress_docs
from app.services.llm_service import generate_response


logger = logging.getLogger(__name__)


def _llm_call(messages: List[Dict[str, str]]) -> str:
    return generate_response(messages).strip()


def _filter_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [doc for doc in docs if float(doc.get("score", 0.0)) > SCORE_THRESHOLD]


def run_agent(query: str, code: str | None = None, session_id: str | None = None) -> Dict[str, Any]:
    del session_id

    start = time.time()
    analysis = analyze_query(query, code)
    pipeline = decide_pipeline(analysis)

    force_retrieval = False
    if analysis.get("language") in {"python", "javascript", "typescript"}:
        force_retrieval = True
    if analysis.get("intent") in {"implementation", "debugging", "concept"}:
        force_retrieval = True

    filters: Dict[str, Any] | None = None
    language = analysis.get("language")
    if language and language != "unknown":
        filters = {"language": language}
    if filters and len(filters) == 0:
        filters = None

    debug_print("[DEBUG ANALYSIS]:", analysis)
    debug_print("[DEBUG FILTERS]:", filters)
    debug_print("[DEBUG FORCE RETRIEVAL]:", force_retrieval)

    code_info = analyze_code_context(code) if code else None

    docs: List[Dict[str, Any]] = []
    should_retrieve = bool(analysis.get("requires_retrieval")) or force_retrieval
    if should_retrieve:
        docs = retriever.retrieve(
            query=query,
            top_k=int(pipeline["top_k"]),
            code_info=code_info,
            filters=filters,
            analysis=analysis if pipeline["use_multi_query"] else None,
        )
        docs = _filter_docs(docs)

        if not docs:
            debug_print("[WARN] Primary retrieval failed -> fallback triggered")
            docs = retriever.hybrid_retrieve(
                query=query,
                top_k=FALLBACK_TOP_K,
                filters=None,
            )
            docs = _filter_docs(docs)
            debug_print("[DEBUG FALLBACK DOCS]:", [doc.get("metadata", {}).get("name") for doc in docs])

        if not docs:
            simplified = " ".join(query.split()[:5])
            docs = retriever.hybrid_retrieve(
                query=simplified,
                top_k=FALLBACK_TOP_K,
                filters=None,
            )
            docs = _filter_docs(docs)
            debug_print("[DEBUG FALLBACK DOCS]:", [doc.get("metadata", {}).get("name") for doc in docs])

    debug_print("[DEBUG DOCS]:", [doc.get("metadata", {}).get("name") for doc in docs])
    debug_print("[DEBUG DOC COUNT]:", len(docs))

    if should_retrieve and not docs:
        debug_print("[PERF] Total time:", time.time() - start)
        return {
            "answer": "I could not find relevant documentation for this query. Please refine your question.",
            "sources": [],
            "analysis": analysis,
        }

    context = compress_docs(query=query, docs=docs)
    prompt = build_prompt(query, context, code, analysis)
    messages = [
        {"role": "system", "content": "You are a senior software engineer helping debug and explain code."},
        {"role": "user", "content": prompt},
    ]

    response = _llm_call(messages)
    validation = validate_response(query, response, context)
    if not validation["is_valid"]:
        response = refine_response(query, response, context, validation["issues"])
    else:
        response = post_process_response(response)

    logger.info(
        "pipeline_complete query=%r docs=%s intent=%s language=%s",
        query,
        len(docs),
        analysis.get("intent"),
        analysis.get("language"),
    )
    debug_print("[PERF] Total time:", time.time() - start)

    return {
        "answer": response,
        "sources": [
            doc.get("source") or doc.get("metadata", {}).get("source")
            for doc in docs
            if doc.get("source") or doc.get("metadata", {}).get("source")
        ],
        "analysis": analysis,
    }


def handle_query(query: str, code: str | None = None) -> str:
    return run_agent(query=query, code=code)["answer"]
