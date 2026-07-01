"""
Centralized OpenRouter GenAI service.
Wraps every call to the OpenRouter API (model: nvidia/nemotron-nano-9b-v2:free) with
caching, latency measurement, call logging (for the evaluation dashboard) and a
safe local fallback when the API key is missing or the request fails.

Note: nvidia/nemotron-3-ultra-550b-a55b:free (the original 550B MoE model) was
measured at 18-148s per call in production logs - unusable for an interactive
UI. Swapped for the 9B nano variant (~6s per call) which is the right
size/latency trade-off for a short justification prompt.
"""
import json
import os
import time
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_MODEL = "nvidia/nemotron-nano-9b-v2:free"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)
LOG_FILE = DATA_DIR / "genai_calls_log.jsonl"


def _log_call(namespace: str, prompt: str, response: str, latency_ms: float,
              cache_hit: bool, fallback: bool, error: Optional[str] = None):
    """Appends one call record to the JSONL log used by the evaluation dashboard."""
    entry = {
        "timestamp": time.time(),
        "namespace": namespace,
        "model": OPENROUTER_MODEL,
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
        print(f"[openrouter_service] Warning: could not write log: {e}")


def call_openrouter(
    prompt: str,
    namespace: str,
    cache: Optional[dict] = None,
    cache_key: Optional[str] = None,
    temperature: float = 0.7,
    fallback_fn=None,
) -> str:
    """
    Calls OpenRouter (nvidia/nemotron-3-ultra:free) with the given prompt, using an
    optional in-memory cache dict (caller owns persistence, e.g. via load_cache()/save_cache()).

    Args:
        prompt: Full prompt text to send to the model.
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
        The generated text (from OpenRouter, cache, or fallback).
    """
    if cache is not None and cache_key is not None and cache_key in cache:
        _log_call(namespace, prompt, cache[cache_key], 0.0, cache_hit=True, fallback=False)
        return cache[cache_key]

    if not OPENROUTER_API_KEY:
        result = fallback_fn() if fallback_fn else ""
        _log_call(namespace, prompt, result, 0.0, cache_hit=False, fallback=True,
                   error="OPENROUTER_API_KEY missing")
        return result

    start = time.perf_counter()
    try:
        response = requests.post(
            OPENROUTER_API_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        text = (data["choices"][0]["message"]["content"] or "").strip()
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
        print(f"[openrouter_service] OpenRouter call failed ({namespace}): {e}")
        return result
