"""
Query enrichment module for user requests (EF4.1).
Automatically enriches short input sentences (< 5 words)
to improve NLP embedding precision.
Uses local Hugging Face models for autonomous processing without API dependency.
"""
import json
from typing import Optional
from pathlib import Path

# Configuration
WORD_THRESHOLD = 5  # Seuil minimum de mots

# Cache pour éviter les appels API répétés
CACHE_FILE = Path(__file__).parent / ".query_cache.json"
_CACHE = None

def load_cache():
    """Loads cache from file."""
    global _CACHE
    if _CACHE is None:
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                    _CACHE = json.load(f)
            except:
                _CACHE = {}
        else:
            _CACHE = {}
    return _CACHE

def save_cache():
    """Saves cache to file."""
    global _CACHE
    if _CACHE is not None:
        try:
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(_CACHE, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving cache: {e}")

def is_query_too_short(query: str) -> bool:
    """
    Checks if a query is too short (< 5 words).
    
    Args:
        query: User query text
        
    Returns:
        True if query contains fewer than 5 words
        
    Example:
        >>> is_query_too_short("action movie")
        True
        >>> is_query_too_short("I want an intense action movie")
        False
    """
    if not query or not query.strip():
        return True
    return len(query.strip().split()) < WORD_THRESHOLD

def augment_query_with_simple_expansion(query: str, use_cache: bool = True) -> str:
    """
    Enriches a short query via simple rule-based approach.
    Lightweight and efficient alternative without heavy model dependency.
    
    Args:
        query: Original user query
        use_cache: Use cache to avoid repeated calls
    
    Returns:
        Enriched query (or original query if already long enough)
    
    Example:
        >>> augment_query_with_simple_expansion("action movie")
        'action movie intense explosive adventure dynamic suspense'
    """
    
    # Check if query is long enough
    if not is_query_too_short(query):
        print(f"[OK] Query long enough ({len(query.split())} words), no enrichment needed")
        return query
    
    # Check cache
    cache = load_cache()
    cache_key = query.lower().strip()
    if use_cache and cache_key in cache:
        print(f"[EF4.1 Cache] - Cache used for '{query}'")
        return cache[cache_key]
    
    print(f"\n[EF4.1] - Short query detected ({len(query.split())} words)")
    print(f"   [Input] : '{query}'")
    print(f"   [Processing] Enriching via context expansion...")
    
    # Dictionary of contextual enrichments (English only for consistency with movies.csv)
    enrichment_map = {
        "action": "action movie intense explosive adventure dynamic thriller combat fight",
        "comedy": "comedy movie humor funny amusing light entertaining hilarious joke",
        "horror": "horror movie fear tension scary dark mature gore psychological",
        "romance": "romance movie love emotional sentimental drama passion feelings",
        "thriller": "thriller movie suspense tension intrigue mystery crime investigation",
        "sci-fi": "science fiction movie futuristic imaginative fantastic technology space",
        "sci-fi movie": "science fiction movie futuristic imaginative fantastic technology space",
        "scifi": "science fiction movie futuristic imaginative fantastic technology space",
        "fantasy": "fantasy movie imaginative magical adventure legendary enchanted",
        "drama": "drama movie intense emotional profound psychological deep",
        "documentary": "documentary movie realistic educational informative historical true",
        "adventure": "adventure movie exploration quest journey discovery exotic",
        "animated": "animation movie animated family colorful creative visual",
        "animation": "animation movie animated family colorful creative visual",
        "mystery": "mystery movie suspense intrigue puzzle investigation secret",
        "war": "war movie conflict military historical combat soldiers",
        "crime": "crime movie criminal investigation detective police justice",
        "family": "family movie children friendly lighthearted wholesome entertainment",
        "western": "western movie frontier cowboy old west adventure classic",
    }
    
    # Find matching keywords
    query_lower = query.lower()
    enriched_parts = [query]
    
    # Search for keyword matches
    for keyword, enrichment in enrichment_map.items():
        if keyword in query_lower:
            enriched_parts.append(enrichment)
            break
    
    # Combine and limit length
    augmented_query = " ".join(enriched_parts)
    words = augmented_query.split()[:20]  # Max 20 words
    augmented_query = " ".join(words)
    
    # Save to cache
    cache[cache_key] = augmented_query
    if use_cache:
        save_cache()
    
    print(f"   [OK] Enrichment successful")
    print(f"   [Output] : '{augmented_query}'")
    print(f"   [Cache] Saved to cache\n")
    return augmented_query

def augment_query_with_gemini(query: str, api_key: Optional[str] = None, use_cache: bool = True) -> str:
    """
    Wrapper for backward compatibility. Now uses simple expansion (EF4.1).
    
    Args:
        query: User query to enrich
        api_key: Not used (backward compatibility)
        use_cache: Use cache to avoid repeated calls
    
    Returns:
        Enriched query
        
    Example:
        >>> augment_query_with_gemini("action movie")
        'action movie intense explosive adventure dynamic suspense'
    """
    return augment_query_with_simple_expansion(query, use_cache)

def augment_query_batch(queries: list[str], api_key: Optional[str] = None) -> list[str]:
    """
    Enriches multiple queries in batch (EF4.1).
    
    Args:
        queries: List of queries to enrich
        api_key: Not used (backward compatibility)
    
    Returns:
        List of enriched queries
    """
    return [augment_query_with_gemini(q, api_key) for q in queries]

