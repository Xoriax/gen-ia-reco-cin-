"""
Evaluation module for generated results (C5.3): metrics and comparisons
consumed by the Streamlit "Évaluation" tab.

Covers:
- distribution of final recommendation scores
- before/after comparison of OpenRouter query enrichment (EF4.1)
- OpenRouter cache hit rate and call latency (from the GenAI call log)
- genre diversity of the recommended set
- side-by-side parameter comparison (temperature) for justification prompts
- a small manual evaluation grid (prompt/response/rating) persisted to CSV
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.services.openrouter_service import call_openrouter, LOG_FILE

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
EVAL_GRID_FILE = DATA_DIR / "genai_eval_grid.csv"


def load_genai_log() -> pd.DataFrame:
    """Loads the OpenRouter call log (prompt/response/latency/cache/fallback per call)."""
    if not LOG_FILE.exists():
        return pd.DataFrame(columns=[
            "timestamp", "namespace", "model", "prompt", "response",
            "latency_ms", "cache_hit", "fallback", "error",
        ])
    rows = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def compute_score_distribution(recommendations: List[Dict]) -> pd.DataFrame:
    """Returns a DataFrame of (title, score) for the current recommendation set."""
    return pd.DataFrame([
        {"title": r.get("title", "?"), "score": r.get("score", 0.0)}
        for r in recommendations
    ])


def compute_before_after_enrichment(recommend_fn, description: str, **kwargs) -> Dict:
    """
    Runs the same request with and without EF4.1 OpenRouter enrichment and compares
    the top result's score, to demonstrate the concrete effect of the GenAI step.

    Args:
        recommend_fn: the `recommend_movies` function (passed in to avoid a
            circular import between recommender and evaluation modules).
        description: free-form description to test.
        kwargs: other recommend_movies parameters (Likert sliders, period, top_k...).
    """
    without = recommend_fn(description=description, enable_augmentation=False, **kwargs)
    with_ = recommend_fn(description=description, enable_augmentation=True, **kwargs)

    without_top = without["recommendations"][0] if without["recommendations"] else None
    with_top = with_["recommendations"][0] if with_["recommendations"] else None

    return {
        "without_enrichment": without_top,
        "with_enrichment": with_top,
        "score_delta": (with_top["score"] - without_top["score"])
        if without_top and with_top else 0.0,
    }


def compute_cache_hit_rate(log_df: Optional[pd.DataFrame] = None) -> Dict:
    """Percentage of OpenRouter calls served from cache vs real API vs fallback."""
    df = log_df if log_df is not None else load_genai_log()
    if df.empty:
        return {"total": 0, "cache_hit_pct": 0.0, "fallback_pct": 0.0, "api_pct": 0.0}

    total = len(df)
    cache_hits = int(df["cache_hit"].sum())
    fallbacks = int(df["fallback"].sum())
    real_api = total - cache_hits - fallbacks

    return {
        "total": total,
        "cache_hit_pct": round(100 * cache_hits / total, 1),
        "fallback_pct": round(100 * fallbacks / total, 1),
        "api_pct": round(100 * real_api / total, 1),
    }


def compute_genre_diversity(recommendations: List[Dict]) -> Dict:
    """Number of unique genres across the recommended set."""
    genres = set()
    for r in recommendations:
        for g in str(r.get("genre", "")).split(","):
            g = g.strip()
            if g:
                genres.add(g)
    return {"unique_genres": len(genres), "genres": sorted(genres)}


def compute_latency_stats(log_df: Optional[pd.DataFrame] = None) -> Dict:
    """Latency stats (ms) for real OpenRouter API calls only (excludes cache/fallback)."""
    df = log_df if log_df is not None else load_genai_log()
    if df.empty:
        return {"count": 0, "mean_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}

    real_calls = df[(~df["cache_hit"]) & (~df["fallback"])]
    if real_calls.empty:
        return {"count": 0, "mean_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}

    return {
        "count": len(real_calls),
        "mean_ms": round(real_calls["latency_ms"].mean(), 1),
        "min_ms": round(real_calls["latency_ms"].min(), 1),
        "max_ms": round(real_calls["latency_ms"].max(), 1),
    }


def compute_temperature_comparison(prompt: str) -> Dict:
    """
    Runs the same prompt with two different OpenRouter temperatures to demonstrate
    a concrete before/after parameter adjustment (grading grid requirement).
    Bypasses the cache on purpose so both calls actually hit the API.
    """
    low = call_openrouter(prompt=prompt, namespace="temperature_comparison_low",
                           temperature=0.2, fallback_fn=lambda: "[fallback] no API key configured")
    high = call_openrouter(prompt=prompt, namespace="temperature_comparison_high",
                            temperature=0.9, fallback_fn=lambda: "[fallback] no API key configured")
    return {
        "temperature_0_2": {"text": low, "length": len(low.split())},
        "temperature_0_9": {"text": high, "length": len(high.split())},
    }


def load_eval_grid() -> pd.DataFrame:
    """Loads the manual evaluation grid (prompt/response/user rating)."""
    if EVAL_GRID_FILE.exists():
        return pd.read_csv(EVAL_GRID_FILE)
    return pd.DataFrame(columns=["timestamp", "namespace", "prompt_excerpt", "response_excerpt", "rating"])


def append_eval_grid_entries(log_df: Optional[pd.DataFrame] = None, n: int = 5) -> pd.DataFrame:
    """
    Seeds/refreshes the manual evaluation grid with the n most recent real
    (non-cache, non-fallback) GenAI calls, ready for the user to rate 1-5
    in the Streamlit UI.
    """
    df = log_df if log_df is not None else load_genai_log()
    grid = load_eval_grid()

    if df.empty:
        return grid

    recent = df[(~df["cache_hit"]) & (~df["fallback"])].tail(n)
    existing_ts = set(grid["timestamp"]) if not grid.empty else set()

    new_rows = []
    for _, row in recent.iterrows():
        if row["timestamp"] in existing_ts:
            continue
        new_rows.append({
            "timestamp": row["timestamp"],
            "namespace": row["namespace"],
            "prompt_excerpt": str(row["prompt"])[:150],
            "response_excerpt": str(row["response"])[:150],
            "rating": None,
        })

    if new_rows:
        grid = pd.concat([grid, pd.DataFrame(new_rows)], ignore_index=True)
        grid.to_csv(EVAL_GRID_FILE, index=False)

    return grid


def save_eval_grid(grid: pd.DataFrame) -> None:
    grid.to_csv(EVAL_GRID_FILE, index=False)
