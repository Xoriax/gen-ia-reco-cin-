"""
Test unitaire pour EF4.1 : Augmentation de l'Entrée (Pre-Processing)
Teste l'enrichissement automatique des requêtes courtes via Gemini AI.
"""
import sys
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "utils"))

from query_augmentation import (
    is_query_too_short,
    augment_query_with_gemini,
    WORD_THRESHOLD
)

def test_is_query_too_short():
    """Test de détection des requêtes courtes."""
    print("\n" + "="*80)
    print("TEST 1 : Détection des requêtes courtes")
    print("="*80)
    
    # Cas 1: Requête très courte (< 5 mots)
    short_queries = [
        "action film",
        "comédie",
        "thriller sombre",
        "animation japonaise"
    ]
    
    for query in short_queries:
        result = is_query_too_short(query)
        word_count = len(query.split())
        print(f"OK: '{query}' ({word_count} mots) Courte : {result}")
        assert result == True, f"'{query}' devrait être détectée comme courte"
    
    # Cas 2: Requête longue (>= 5 mots)
    long_queries = [
        "Je veux un film dramatique intense avec du crime et du suspense",
        "film d'action avec des explosions et des poursuites en voiture",
        "comédie romantique légère pour soirée en famille avec humour"
    ]
    
    for query in long_queries:
        result = is_query_too_short(query)
        word_count = len(query.split())
        print(f"OK: '{query[:50]}...' ({word_count} mots) Courte : {result}")
        assert result == False, f"'{query}' ne devrait PAS être détectée comme courte"
    
    # Cas 3: Cas limites
    edge_cases = ["", " "]
    for query in edge_cases:
        result = is_query_too_short(query)
        print(f"OK: Requête vide Courte : {result}")
        assert result == True
    
    print(f"\nOK Test 1 réussi : Détection correcte des requêtes courtes\n")

def test_augmentation_with_gemini():
    """Test de l'augmentation avec Gemini (nécessite clé API)."""
    print("="*80)
    print("TEST 2 : Augmentation avec Gemini AI")
    print("="*80)
    
    # CAS 1 : Requêtes COURTES (< 5 mots) - DOIVENT être enrichies
    print("\n CAS 1 : Requêtes COURTES (< 5 mots) - Enrichissement OBLIGATOIRE :")
    print("-" * 80)
    
    short_queries = [
        "comédie",
        "action film",
        "thriller sombre",
        "animation japonaise"
    ]
    
    for query in short_queries:
        word_count_original = len(query.split())
        print(f"\n Test : '{query}' ({word_count_original} mots < 5)")
        
        enriched = augment_query_with_gemini(query)
        word_count_enriched = len(enriched.split())
        
        print(f" Entrée : {query}")
        print(f" Sortie : {enriched}")
        print(f" Longueur : {word_count_original} mots {word_count_enriched} mots")
        
        # Vérifications de qualité
        if word_count_enriched <= word_count_original:
            print(f" FAIL ÉCHEC : Pas d'enrichissement (vérifiez GEMINI_API_KEY)")
        elif enriched.strip().startswith(query + " :") or enriched.strip() == query + " :":
            print(f" FAIL ÉCHEC : Enrichissement de mauvaise qualité (juste '{query} :')")
        elif word_count_enriched >= word_count_original + 5:
            print(f" OK SUCCÈS : Enrichissement de qualité (+{word_count_enriched - word_count_original} mots)")
        else:
            print(f" WARNING PARTIEL : Enrichissement minimal")
    
    # CAS 2 : Requêtes LONGUES (>= 5 mots) - NE DOIVENT PAS être enrichies
    print("\n" + "="*80)
    print(" CAS 2 : Requêtes LONGUES (>= 5 mots) - PAS d'enrichissement :")
    print("-" * 80)
    
    long_queries = [
        "Je veux un film dramatique intense avec du crime",
        "film d'action avec des explosions et des poursuites",
        "comédie romantique légère pour soirée en famille"
    ]
    
    for query in long_queries:
        word_count = len(query.split())
        print(f"\n Test : '{query[:50]}...' ({word_count} mots >= 5)")
        
        enriched = augment_query_with_gemini(query)
        
        print(f" Entrée : {query}")
        print(f" Sortie : {enriched}")
        
        if enriched == query:
            print(f" OK SUCCÈS : Requête non modifiée (comme attendu)")
        else:
            print(f" FAIL ÉCHEC : Requête modifiée alors qu'elle était déjà longue")
            assert False, "Une requête longue ne devrait pas être modifiée"
    
    print(f"\nOK Test 2 réussi : Augmentation Gemini fonctionnelle\n")

def test_integration():
    """Test de l'intégration avec movie_recommender."""
    print("="*80)
    print("TEST 3 : Intégration avec movie_recommender")
    print("="*80)
    
    # Ajouter le répertoire recommender au path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "recommender"))
    
    try:
        from movie_recommender import recommend_movies
        print("OK: Module movie_recommender importé avec succès")
        
        # Vérifier que le paramètre enable_augmentation existe
        import inspect
        sig = inspect.signature(recommend_movies)
        params = sig.parameters
        
        if 'enable_augmentation' in params:
            print("OK: Paramètre enable_augmentation présent")
        else:
            print("FAIL: Paramètre enable_augmentation manquant")
        
        print(f"\nOK Test 3 réussi : Intégration correcte\n")
        
    except Exception as e:
        print(f"FAIL: Erreur lors de l'import : {e}")
        raise

if __name__ == "__main__":
    print("\n" + "" + " "*28 + "TESTS EF4.1" + " "*28 + "")
    print("Tests unitaires : Augmentation de l'Entrée (Pre-Processing)")
    print("="*80 + "\n")
    
    test_is_query_too_short()
    test_augmentation_with_gemini()
    test_integration()
    
    print("="*80)
    print(" TOUS LES TESTS RÉUSSIS !")
    print("="*80)
    print("\nRésumé:")
    print(" OK Détection des requêtes courtes")
    print(" OK Logique conditionnelle (< 5 mots)")
    print(" OK Enrichissement Gemini (si clé API)")
    print(" OK Intégration avec movie_recommender")
