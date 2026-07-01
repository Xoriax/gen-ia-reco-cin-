"""
Centralized Gemini GenAI service.
Wraps every call to the Gemini API with caching, latency measurement,
call logging (for the evaluation dashboard) and a safe local fallback
when the API key is missing or the request fails.
"""
import json
import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)
LOG_FILE = DATA_DIR / "genai_calls_log.jsonl"

_client = None


def _get_client():
    """Lazily create the Gemini client (avoids import/network cost when unused)."""
    global _client
    if _client is None and GEMINI_API_KEY:
        from google import genai
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def _log_call(namespace: str, prompt: str, response: str, latency_ms: float,
              cache_hit: bool, fallback: bool, error: Optional[str] = None):
    """Appends one call record to the JSONL log used by the evaluation dashboard."""
    entry = {
        "timestamp": time.time(),
        "namespace": namespace,
        "model": GEMINI_MODEL,
        "prompt": prompt,
        "response": response,
        "latency_ms": latency_ms,
        "cache_hit": cache_hit,
        "fallback": fallback,
        "error": error,
    }
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[gemini_service] Warning: could not write log: {e}")


def call_gemini(
    prompt: str,
    namespace: str,
    cache: Optional[dict] = None,
    cache_key: Optional[str] = None,
    temperature: float = 0.7,
    fallback_fn=None,
) -> str:
    """
    Calls Gemini with the given prompt, using an optional in-memory cache dict
    (caller owns persistence, e.g. via load_cache()/save_cache()).

    Args:
        prompt: Full prompt text to send to Gemini.
        namespace: Logical usage name (e.g. "query_augmentation", "justification"),
            used for logging/evaluation grouping.
        cache: Optional dict used as a cache (key -> response).
        cache_key: Key to look up/store in `cache`.
        temperature: Sampling temperature (used for the parameter-tuning comparison
            in the evaluation dashboard).
        fallback_fn: Zero-arg callable returning a local rule-based result if the
            API key is missing or the call fails. Required so the app never crashes
            without network access / quota.

    Returns:
        The generated text (from Gemini, cache, or fallback).
    """
    if cache is not None and cache_key is not None and cache_key in cache:
        _log_call(namespace, prompt, cache[cache_key], 0.0, cache_hit=True, fallback=False)
        return cache[cache_key]

    client = _get_client()
    if client is None:
        result = fallback_fn() if fallback_fn else ""
        _log_call(namespace, prompt, result, 0.0, cache_hit=False, fallback=True,
                   error="GEMINI_API_KEY missing")
        return result

    start = time.perf_counter()
    try:
        from google.genai import types
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=temperature),
        )
        text = (response.text or "").strip()
        latency_ms = (time.perf_counter() - start) * 1000

        if cache is not None and cache_key is not None:
            cache[cache_key] = text

        _log_call(namespace, prompt, text, latency_ms, cache_hit=False, fallback=False)
        return text
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        result = fallback_fn() if fallback_fn else ""
        _log_call(namespace, prompt, result, latency_ms, cache_hit=False, fallback=True,
                   error=str(e))
        print(f"[gemini_service] Gemini call failed ({namespace}): {e}")
        return result
