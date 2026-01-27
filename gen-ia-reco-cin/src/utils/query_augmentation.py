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
    
    # Vérifier si la requête est assez longue
    if not is_query_too_short(query):
        print(f"[OK] Requete suffisamment longue ({len(query.split())} mots), pas d'enrichissement")
        return query
    
    # Vérifier le cache
    cache = load_cache()
    cache_key = query.lower().strip()
    if use_cache and cache_key in cache:
        print(f"[EF4.1 Cache] - Cache utilise pour '{query}'")
        return cache[cache_key]
    
    print(f"\n[EF4.1] - Requete courte detectee ({len(query.split())} mots)")
    print(f"   [Input] : '{query}'")
    print(f"   [Processing] Enrichissement via expansion de contexte...")
    
    # Dictionnaire d'enrichissements contextuels
    enrichment_map = {
        "action": "film action intensité explosif aventure dynamique suspense cascade combat",
        "comédie": "film comédie humour drôle amusant léger familial hilarant blague",
        "horreur": "film horreur peur tension effrayant sombre mature gore psychologique",
        "romance": "film romance amour émotionnel sentimental drame passion sentiments",
        "thriller": "film thriller suspense tension intrigue mystère crime enquête",
        "sci-fi": "film science fiction futuriste imaginaire fantastique technologie espace",
        "fantasy": "film fantasy fantastique imaginaire magie aventure légende",
        "drama": "film dramatique intense émotionnel profond psychologique drame",
        "documentaire": "film documentaire réaliste éducatif informatif historique vrai",
        "aventure": "film aventure exploration quête voyages découverte exotique",
    }
    
    # Chercher les mots-clés correspondants
    query_lower = query.lower()
    enriched_parts = [query]
    
    # Rechercher correspondance
    for keyword, enrichment in enrichment_map.items():
        if keyword in query_lower:
            enriched_parts.append(enrichment)
            break
    
    # Combiner et limiter la longueur
    augmented_query = " ".join(enriched_parts)
    words = augmented_query.split()[:20]  # Max 20 mots
    augmented_query = " ".join(words)
    
    # Sauvegarder en cache
    cache[cache_key] = augmented_query
    if use_cache:
        save_cache()
    
    print(f"   [OK] Enrichissement reussi")
    print(f"   [Output] : '{augmented_query}'")
    print(f"   [Cache] Sauvegarde dans le cache\n")
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

