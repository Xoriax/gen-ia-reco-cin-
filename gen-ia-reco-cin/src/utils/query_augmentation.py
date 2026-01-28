"""
Module d'enrichissement de requêtes utilisateur (EF4.1).
Enrichit automatiquement les phrases d'entrée trop courtes (< 5 mots)
pour améliorer la précision des embeddings NLP.
Utilise des modèles locaux Hugging Face pour un traitement autonome sans dépendance API.
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
    """Charge le cache depuis le fichier."""
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
    """Sauvegarde le cache dans le fichier."""
    global _CACHE
    if _CACHE is not None:
        try:
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(_CACHE, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Erreur lors de la sauvegarde du cache : {e}")

def is_query_too_short(query: str) -> bool:
    """
    Vérifie si une requête est trop courte (< 5 mots).
    
    Args:
        query: Texte de la requête utilisateur
        
    Returns:
        True si la requête contient moins de 5 mots
        
    Example:
        >>> is_query_too_short("action film")
        True
        >>> is_query_too_short("Je veux un film d'action intense")
        False
    """
    if not query or not query.strip():
        return True
    return len(query.strip().split()) < WORD_THRESHOLD

def augment_query_with_simple_expansion(query: str, use_cache: bool = True) -> str:
    """
    Enrichit une requête courte via une approche simple basée sur des règles.
    Alternative légère et efficace sans dépendance modèle lourd.
    
    Args:
        query: Requête utilisateur originale
        use_cache: Utiliser le cache pour éviter les appels répétés
    
    Returns:
        Requête enrichie (ou requête originale si déjà suffisamment longue)
    
    Example:
        >>> augment_query_with_simple_expansion("action film")
        'action film intensité explosif aventure dramatique suspense'
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
    Wrapper pour rétrocompatibilité. Utilise désormais l'expansion simple (EF4.1).
    
    Args:
        query: Requête utilisateur à enrichir
        api_key: Non utilisé (rétrocompatibilité)
        use_cache: Utiliser le cache pour éviter les appels répétés
    
    Returns:
        Requête enrichie
        
    Example:
        >>> augment_query_with_gemini("action film")
        'action film intensité explosif aventure dynamique suspense'
    """
    return augment_query_with_simple_expansion(query, use_cache)

def augment_query_batch(queries: list[str], api_key: Optional[str] = None) -> list[str]:
    """
    Enrichit plusieurs requêtes en batch (EF4.1).
    
    Args:
        queries: Liste de requêtes à enrichir
        api_key: Non utilisé (rétrocompatibilité)
    
    Returns:
        Liste des requêtes enrichies
    """
    return [augment_query_with_gemini(q, api_key) for q in queries]

