from __future__ import annotations

from typing import Any, Dict


FEW_SHOT_EXAMPLE = """EXAMPLE:
USER:
"why is async slow python"

ASSISTANT:
## Explanation
Using `time.sleep()` inside an async function blocks the event loop.

## Fix
Replace `time.sleep()` with `await asyncio.sleep()`.

## Example
```python
import asyncio

async def foo():
    await asyncio.sleep(1)
```
"""


def _render_context(context: Any) -> str:
    if isinstance(context, dict):
        lines = []
        for section in ("concepts", "fixes", "examples"):
            values = context.get(section) or []
            if not values:
                continue
            lines.append(f"{section.title()}:")
            lines.extend(str(value) for value in values)
        return "\n".join(lines).strip()
    return str(context or "").strip()


def build_prompt(query: str, context: Any, code: str = None, analysis: Dict[str, Any] = None) -> str:
    analysis = analysis or {}
    intent = analysis.get("intent", "general")
    rendered_context = _render_context(context) or "No relevant documentation retrieved"

    behavior_lines = [
        "- Use ONLY the provided CONTEXT to answer",
        '- If context is insufficient, say "I don\'t have enough information"',
        "- Do NOT hallucinate",
        "- Be concise and practical",
        "- Prefer code examples when relevant",
        "- Reference relevant concepts from CONTEXT when possible",
    ]

    if intent == "debugging":
        behavior_lines.append("- Focus on identifying root cause and fixing the bug")
    elif intent == "implementation":
        behavior_lines.append("- Provide step-by-step implementation guidance")
    elif intent == "concept":
        behavior_lines.append("- Explain clearly with simple examples")

    if code:
        behavior_lines.append("- Prioritize the code fix over theory and keep the explanation short")

    if not rendered_context or rendered_context == "No relevant documentation retrieved":
        behavior_lines.append("WARNING: No external documentation found. Answer cautiously.")

    return f"""ROLE:
You are a senior software engineer helping debug and explain code.

{FEW_SHOT_EXAMPLE}

INSTRUCTIONS:
{chr(10).join(behavior_lines)}

CONTEXT:
{rendered_context}

USER QUERY:
{query}

CODE (if provided):
{code if code else "No code provided"}

OUTPUT FORMAT:

## Explanation
- Root cause
- Key concept

## Fix
- Clear actionable fix

## Example
- Minimal working code snippet
"""
