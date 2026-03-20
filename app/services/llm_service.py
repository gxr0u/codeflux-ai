from __future__ import annotations

from typing import Dict, List

from groq import Groq
from dotenv import load_dotenv

from app.config.debug import debug_print
from app.config.settings import GROQ_API_KEY, LLM_MAX_TOKENS, LLM_TEMPERATURE, MODEL_NAME


load_dotenv()

if GROQ_API_KEY is None:
    raise ValueError("Missing GROQ_API_KEY")


client = Groq(api_key=GROQ_API_KEY)


def generate_response(messages: List[Dict]) -> str:
    try:
        response = client.chat.completions.create(
            messages=messages,
            model=MODEL_NAME,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        return response.choices[0].message.content
    except Exception as exc:
        debug_print("[ERROR] LLM failed:", exc)
        return "The system encountered an error while generating a response."
