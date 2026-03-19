# ⚡ CodeFlux

Agentic AI coding assistant powered by advanced RAG and context-aware reasoning.

CodeFlux is a production-style AI system that understands, debugs, and explains code using a retrieval-first architecture instead of relying purely on LLM memory.

Unlike traditional copilots, CodeFlux ensures grounded, accurate, and structured responses by combining:

- Hybrid Retrieval (FAISS + BM25 + Multi-Query)
- Query Analysis & Dynamic Routing
- Groq LLM (fast inference)
- Context Compression & Reranking
- Agentic Pipeline for reasoning

--------------------------------------------------

FEATURES

- Debugs code with root-cause analysis
- Uses grounded documentation (RAG) instead of hallucinating
- Multi-query retrieval for better recall
- Hybrid search (semantic + keyword)
- Structured responses (Explanation / Fix / Example)
- Language-aware filtering (Python, JS, TS)
- Fallback mechanisms (no silent failures)
- Production-style modular architecture

--------------------------------------------------

ARCHITECTURE

Query
 ↓
Query Analyzer
 ↓
Dynamic Pipeline Routing
 ↓
Multi-Query + Hybrid Retrieval
 ↓
Reranker + Deduplication
 ↓
Context Compression
 ↓
Prompt Builder
 ↓
Groq LLM
 ↓
Validated Response

--------------------------------------------------

EXAMPLE

Input:
why is my async code slow in python

Output:

Explanation
- Root cause: Using time.sleep() inside an async function blocks the event loop.

Fix
- Replace time.sleep() with await asyncio.sleep()

Example
import asyncio
async def foo():
    await asyncio.sleep(1)

--------------------------------------------------

SETUP

git clone https://github.com/your-username/codeflux-ai
cd codeflux-ai

python -m venv .venv
.venv\Scripts\activate   (Windows)
or
source .venv/bin/activate   (Mac/Linux)

pip install -r requirements.txt

--------------------------------------------------

ENVIRONMENT VARIABLES

Create a .env file:

GROQ_API_KEY=your_api_key_here

--------------------------------------------------

INGEST DOCUMENTATION

python -m app.rag.ingest

--------------------------------------------------

RUN

python -m app.test_pipeline

--------------------------------------------------

TECH STACK

- Python
- FAISS
- Sentence Transformers (BGE embeddings)
- Groq (LLM inference)
- Custom RAG pipeline

--------------------------------------------------

FUTURE WORK

- FastAPI backend
- React + Monaco editor frontend
- Multi-turn memory
- Observability & tracing
- Deployment (Docker + cloud)

--------------------------------------------------

WHY CODEFLUX?

Most AI coding tools rely heavily on LLM memory.

CodeFlux is different:
- It retrieves first, generates second
- It forces grounding
- It behaves like a reasoning system, not just autocomplete

--------------------------------------------------

AUTHOR

Aditya Verma
