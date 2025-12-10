"""
Module d'enrichissement de requêtes utilisateur via Gemini AI (EF4.1).
Enrichit automatiquement les phrases d'entrée trop courtes (< 5 mots)
pour améliorer la précision des embeddings NLP.
Inclut un cache pour éviter les appels API répétés.
"""
import os
import requests
import json
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
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

def augment_query_with_gemini(query: str, api_key: Optional[str] = None, use_cache: bool = True) -> str:
    """
    Enrichit une requête courte via l'API Gemini.
    
    Si la requête contient >= 5 mots, elle est retournée sans modification.
    Sinon, Gemini ajoute du contexte technique et descriptif.
    Utilise un cache pour éviter les appels API répétés (erreur 429).
    
    Args:
        query: Requête utilisateur à enrichir
        api_key: Clé API Gemini (optionnelle, utilise GEMINI_API_KEY par défaut)
        use_cache: Utiliser le cache pour éviter les appels répétés
    
    Returns:
        Requête enrichie ou originale si >= 5 mots
        
    Example:
        >>> augment_query_with_gemini("action film")
        "film d'action intense avec des scènes de combat dynamiques, 
         des poursuites spectaculaires, des explosions et un rythme soutenu"
    """
    # Vérifier si l'augmentation est nécessaire
    if not is_query_too_short(query):
        print(f"✅ Requête longue ({len(query.split())} mots) - Pas d'enrichissement nécessaire")
        return query
    
    print(f"\n🔍 EF4.1 - Requête courte détectée ({len(query.split())} mots)")
    print(f"   📝 Entrée : '{query}'")
    
    # Vérifier le cache
    if use_cache:
        cache = load_cache()
        cache_key = query.lower().strip()
        if cache_key in cache:
            enriched = cache[cache_key]
            print(f"   📦 Cache utilisé")
            print(f"   ✨ Sortie : '{enriched[:80]}...'")
            return enriched
    
    # Récupérer la clé API
    key = api_key or GEMINI_API_KEY
    if not key:
        print("   ⚠️ GEMINI_API_KEY non définie, requête non enrichie")
        return query
    
    print(f"   🌐 Appel API Gemini en cours...")
    
    try:
        # Construire le prompt d'enrichissement
        prompt = f"""Tu es un expert en cinéma et séries TV. Un utilisateur recherche un film/série avec cette description très courte :

"{query}"

Enrichis cette description en ajoutant du contexte technique et descriptif pour améliorer la recherche. 

Instructions :
1. Garde l'intention originale de l'utilisateur
2. Ajoute des détails sur l'atmosphère, le style, le rythme, les thèmes
3. Reste concis (2-3 phrases maximum)
4. Utilise un langage descriptif riche
5. Ne réponds QUE avec la description enrichie, sans explication

Exemple :
Entrée: "action film"
Sortie: "Film d'action avec des scènes de combat spectaculaires, des poursuites automobiles, des explosions, un rythme intense et dynamique avec une atmosphère adrénaline"

Enrichis maintenant: "{query}"
"""
        
        # Appel API REST
        headers = {
            'Content-Type': 'application/json',
            'x-goog-api-key': key
        }
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 150
            }
        }
        
        response = requests.post(GEMINI_API_URL, json=payload, headers=headers, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            
            # Extraire le texte de la réponse
            if 'candidates' in result and len(result['candidates']) > 0:
                candidate = result['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    enriched_text = candidate['content']['parts'][0].get('text', '').strip()
                    
                    if enriched_text and len(enriched_text) > len(query):
                        print(f"   ✅ Enrichissement réussi")
                        print(f"   ✨ Sortie : '{enriched_text[:80]}...'")
                        
                        # Sauvegarder dans le cache
                        if use_cache:
                            cache = load_cache()
                            cache[query.lower().strip()] = enriched_text
                            save_cache()
                            print(f"   💾 Sauvegardé dans le cache")
                        
                        return enriched_text
            
            print(f"   ⚠️ Réponse Gemini invalide, utilisation de la requête originale")
            return query
            
        elif response.status_code == 429:
            print(f"   ⚠️ Quota API Gemini dépassé (429)")
            print(f"   💡 Attendez quelques minutes ou utilisez le cache")
            print(f"   💡 Quota gratuit : 15 requêtes/minute, 1500/jour")
            return query
        else:
            print(f"   ⚠️ Erreur API Gemini ({response.status_code})")
            return query
            
    except requests.exceptions.Timeout:
        print(f"   ⚠️ Timeout API Gemini")
        return query
    except Exception as e:
        print(f"   ⚠️ Erreur : {e}")
        return query

def augment_query_batch(queries: list[str], api_key: Optional[str] = None) -> list[str]:
    """
    Enrichit plusieurs requêtes en batch.
    
    Args:
        queries: Liste de requêtes à enrichir
        api_key: Clé API Gemini (optionnelle)
    
    Returns:
        Liste des requêtes enrichies
    """
    return [augment_query_with_gemini(q, api_key) for q in queries]
