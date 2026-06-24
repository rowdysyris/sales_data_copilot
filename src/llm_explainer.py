from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


SYSTEM_RULES = (
    "You are a Senior Data Analyst with 10 years experience. "
    "You will receive a user question, the detected intent, and only the relevant pre-aggregated data. "
    "If question is ambiguous interpret the most likely business intent and state your interpretation upfront. "
    "If user assumes something wrong correct them politely with data — never agree with wrong assumptions. "
    "If question has multiple intents answer all of them. If question is a follow up use context to answer. "
    "Never say the question is too vague — always attempt answer. Never return generic stats — always answer what was asked. "
    "Always use $ never ₹. Always end with a recommendation or action item. "
    "If a term is non standard explain what you interpreted it as. "
    "Use this answer format: INTERPRETATION → ANSWER → DATA → INSIGHT → RECOMMENDATION. "
    "Do not invent, alter, estimate, or change any numbers. All numeric claims must remain exactly as provided by the deterministic Pandas analysis."
)


def _provider() -> str:
    return os.getenv("LLM_PROVIDER", "none").strip().lower()


def _ollama_generate(prompt: str) -> str | None:
    base_url = os.getenv("OLLAMA_BASE_URL", os.getenv("OLLAMA_API_URL", "http://localhost:11434")).rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "llama3.1")
    payload = json.dumps({
        "model": model,
        "prompt": f"{SYSTEM_RULES}\n\nAnalysis to rewrite:\n{prompt}",
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=18) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("response") or None
    except Exception:
        return None


def _extract_responses_text(data: dict[str, Any]) -> str | None:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    output = data.get("output", [])
    pieces: list[str] = []
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []) or []:
                if isinstance(content, dict):
                    text = content.get("text") or content.get("output_text")
                    if isinstance(text, str):
                        pieces.append(text)
    return "\n".join(pieces).strip() or None


def _openai_generate(prompt: str) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    payload = json.dumps({
        "model": model,
        "instructions": SYSTEM_RULES,
        "input": prompt,
        "temperature": 0.1,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return _extract_responses_text(data)
    except Exception:
        return None


def beautify_answer_with_llm(markdown_answer: str, computed_context: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    """Optionally improve wording with OpenAI/Ollama while preserving computed evidence.

    The raw dataframe is never sent to the LLM. Only the deterministic answer plus compact
    metadata is sent. If the provider is disabled or fails, the deterministic answer is returned.
    """
    provider = _provider()
    meta = {"provider": provider, "used": False, "fallback_used": True}
    if provider in {"", "none", "disabled", "off"}:
        meta["reason"] = "LLM_PROVIDER is none."
        return markdown_answer, meta

    compact = ""
    if computed_context:
        compact_bits = {
            "confidence": computed_context.get("confidence"),
            "limitations": computed_context.get("limitations", []),
            "relationship_findings": computed_context.get("relationship_findings", [])[:8],
            "root_causes": computed_context.get("root_causes", [])[:5],
        }
        compact = "\n\nComputed context:\n" + json.dumps(compact_bits, default=str, ensure_ascii=False)
    prompt = markdown_answer + compact

    improved = None
    if provider == "ollama":
        improved = _ollama_generate(prompt)
    elif provider == "openai":
        improved = _openai_generate(prompt)
    else:
        meta["reason"] = f"Unsupported provider: {provider}"
        return markdown_answer, meta

    if improved and isinstance(improved, str) and len(improved.strip()) > 50:
        meta.update({"used": True, "fallback_used": False})
        return improved.strip(), meta
    meta["reason"] = "LLM unavailable or returned insufficient content."
    return markdown_answer, meta
