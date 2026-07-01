"""
GenAI module for generating personalized recommendation justification text (EF4.3).
Calls Gemini once per recommendation set to produce a narrative synthesis
explaining why the top films match the user's cinematic profile. The prompt is
grounded in the already-selected top-3 titles (from SBERT scoring) so Gemini
only comments on them instead of inventing new recommendations (mitigates
hallucination risk).
"""
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.services.gemini_service import call_gemini

CACHE_FILE = Path(__file__).parent / ".justification_cache.json"
_CACHE = None


def load_cache():
    global _CACHE
    if _CACHE is None:
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                    _CACHE = json.load(f)
            except Exception:
                _CACHE = {}
        else:
            _CACHE = {}
    return _CACHE


def save_cache():
    global _CACHE
    if _CACHE is not None:
        try:
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(_CACHE, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: could not save justification cache: {e}")


def _build_prompt(user_preferences: Dict, recommendations: List[Dict],
                   priority_elements: Optional[List[str]]) -> str:
    top = recommendations[:3]
    titles_block = "\n".join(
        f"- \"{r.get('title', 'Unknown')}\" ({r.get('year', '?')}, {r.get('genre', 'N/A')}) "
        f"- match score: {r.get('score', 0):.1%}"
        for r in top
    )
    refinement = ""
    if priority_elements:
        refinement = f"\nLeast-weighted scoring factors (could be refined further): {', '.join(priority_elements)}."

    return (
        "You are a film recommendation assistant. A semantic search engine (SBERT) already "
        "selected the movies below for this user - do NOT suggest any other film, only "
        "explain and contextualize these exact results.\n\n"
        "User profile:\n"
        f"- Free-form request: \"{user_preferences.get('description', '')}\"\n"
        f"- Similar title mentioned: \"{user_preferences.get('similar_title', '')}\"\n"
        f"- Action intensity (1-5): {user_preferences.get('action_intensity', 3)}\n"
        f"- Narrative complexity (1-5): {user_preferences.get('narrative_complexity', 3)}\n"
        f"- Darkness/tone (1-5): {user_preferences.get('darkness', 3)}\n"
        f"- Realism vs fantasy (1-5): {user_preferences.get('realism', 3)}\n"
        f"- Preferred period: \"{user_preferences.get('period', 'any')}\"\n\n"
        f"Top recommended films:\n{titles_block}\n"
        f"{refinement}\n\n"
        "Write a short, engaging justification (3-5 sentences) explaining why the top film "
        "fits this profile, briefly mentioning the second choice as an alternative, in English."
    )


def _cache_key(user_preferences: Dict, recommendations: List[Dict]) -> str:
    top_titles = [r.get("title", "") for r in recommendations[:3]]
    raw = json.dumps({"prefs": user_preferences, "titles": top_titles}, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _local_fallback(recommendations: List[Dict]) -> str:
    """Minimal local fallback used only if Gemini is unavailable."""
    if not recommendations:
        return ""
    top = recommendations[0]
    return (
        f"Based on your preferences, \"{top.get('title', 'this film')}\" is our top match "
        f"with a {top.get('score', 0):.1%} affinity score."
    )


def generate_recommendation_justification(
    user_preferences: Dict,
    recommendations: List[Dict],
    priority_elements: List[str] = None
) -> Optional[str]:
    """
    Generate a personalized recommendation justification via Gemini (EF4.3).
    One API call per recommendation set (cached on preferences + top titles).

    Args:
        user_preferences: User's complete search criteria
        recommendations: Top movie recommendations with scores
        priority_elements: Low-scoring elements for future refinement

    Returns:
        Generated narrative justification, or None if there is nothing to justify.
    """
    if not recommendations or len(recommendations) < 1:
        return None

    try:
        cache = load_cache()
        key = _cache_key(user_preferences, recommendations)
        prompt = _build_prompt(user_preferences, recommendations, priority_elements)

        justification = call_gemini(
            prompt=prompt,
            namespace="justification",
            cache=cache,
            cache_key=key,
            temperature=0.7,
            fallback_fn=lambda: _local_fallback(recommendations),
        )
        save_cache()
        return justification or _local_fallback(recommendations)
    except Exception as e:
        print(f"Error in EF4.3 justification generation: {e}")
        return _local_fallback(recommendations)
