"""Utility for generating concise web-search queries using an LLM via the `ollama` Python SDK.

Uses the lightweight `qwen2.5-0.5b-instruct` model locally (or proxied) through Ollama.  If that model isn’t present, you may set `LLM_QUERY_MODEL` env var to another small instruct model such as `deepseek-r1:1.5b`.   This keeps the
logic server-side so we don’t rely on a third-party provider.
"""
from functools import lru_cache
from typing import Optional

try:
    import ollama  # type: ignore
except ImportError as e:  # pragma: no cover – runtime import guard
    raise RuntimeError(
        "ollama package is required for llm_query_service – install and ensure Ollama daemon is running"
    ) from e

# Model to use for query generation – can be over-ridden with env var if desired
import os
MODEL_NAME = os.getenv("LLM_QUERY_MODEL", "qwen2.5-0.5b-instruct")

_SYSTEM_PROMPT = (
    "You are an expert at converting a natural-language question into a succinct web-search query. "
    "Return ONLY the search query. Do NOT include explanations or punctuation."
)


@lru_cache(maxsize=1024)
def generate_search_query(prompt: str) -> str:
    """Generate a search query string from an arbitrary user prompt.

    This is a synchronous helper – internally cached – because the same prompts are often repeated
    as we refine context.  Raises `RuntimeError` if Ollama returns an unexpected response.
    """
    # Guard against empty prompt
    if not prompt or not prompt.strip():
        return ""

    try:
        # If the preferred model isn’t available, Ollama will raise an error – callers can set the env var accordingly.
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt.strip()},
            ],
            temperature=0.2,
            top_p=0.9,
        )
        query: Optional[str] = response.get("message", {}).get("content") if isinstance(response, dict) else None
        if not query:
            raise RuntimeError("No content field in Ollama response")
        return query.strip().strip("\n")
    except Exception as exc:
        # Log and fall back to using the raw prompt
        import logging

        logging.getLogger(__name__).error("LLM query generation failed: %s", exc)
        return prompt.strip()  # Fallback – at least not worse than before
