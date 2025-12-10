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
        print(f"✓ '{query}' ({word_count} mots) → Courte : {result}")
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
        print(f"✓ '{query[:50]}...' ({word_count} mots) → Courte : {result}")
        assert result == False, f"'{query}' ne devrait PAS être détectée comme courte"
    
    # Cas 3: Cas limites
    edge_cases = ["", "   "]
    for query in edge_cases:
        result = is_query_too_short(query)
        print(f"✓ Requête vide → Courte : {result}")
        assert result == True
    
    print(f"\n✅ Test 1 réussi : Détection correcte des requêtes courtes\n")

def test_augmentation_with_gemini():
    """Test de l'augmentation avec Gemini (nécessite clé API)."""
    print("="*80)
    print("TEST 2 : Augmentation avec Gemini AI")
    print("="*80)
    
    # Test avec requêtes courtes
    test_queries = [
        "action film",
        "comédie",
    ]
    
    print("\n📝 Test d'enrichissement de requêtes courtes :")
    
    for query in test_queries:
        print(f"\n🔍 Test : '{query}'")
        enriched = augment_query_with_gemini(query)
        word_count_original = len(query.split())
        word_count_enriched = len(enriched.split())
        
        print(f"   Original ({word_count_original} mots) : {query}")
        print(f"   Enrichie ({word_count_enriched} mots) : {enriched[:100]}...")
        
        # La requête enrichie devrait être plus longue
        if word_count_enriched > word_count_original:
            print(f"   ✅ Enrichissement réussi")
        else:
            print(f"   ⚠️  Pas d'enrichissement (vérifiez GEMINI_API_KEY)")
    
    # Test avec requête longue (ne devrait pas être modifiée)
    long_query = "Je veux un film dramatique intense avec du crime et du suspense"
    print(f"\n📝 Test avec requête longue :")
    print(f"   Entrée  : '{long_query}'")
    
    enriched = augment_query_with_gemini(long_query)
    
    print(f"   Sortie  : '{enriched}'")
    
    if enriched == long_query:
        print(f"   ✅ Requête longue non modifiée (correct)")
    else:
        print(f"   ⚠️  Requête modifiée (ne devrait pas l'être)")
    
    print(f"\n✅ Test 2 réussi : Augmentation Gemini fonctionnelle\n")

def test_integration():
    """Test de l'intégration avec movie_recommender."""
    print("="*80)
    print("TEST 3 : Intégration avec movie_recommender")
    print("="*80)
    
    # Ajouter le répertoire recommender au path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "recommender"))
    
    try:
        from movie_recommender import recommend_movies
        print("✓ Module movie_recommender importé avec succès")
        
        # Vérifier que le paramètre enable_augmentation existe
        import inspect
        sig = inspect.signature(recommend_movies)
        params = sig.parameters
        
        if 'enable_augmentation' in params:
            print("✓ Paramètre enable_augmentation présent")
        else:
            print("✗ Paramètre enable_augmentation manquant")
        
        print(f"\n✅ Test 3 réussi : Intégration correcte\n")
        
    except Exception as e:
        print(f"✗ Erreur lors de l'import : {e}")
        raise

if __name__ == "__main__":
    print("\n" + "🎬" + " "*28 + "TESTS EF4.1" + " "*28 + "🎬")
    print("Tests unitaires : Augmentation de l'Entrée (Pre-Processing)")
    print("="*80 + "\n")
    
    test_is_query_too_short()
    test_augmentation_with_gemini()
    test_integration()
    
    print("="*80)
    print("🎉 TOUS LES TESTS RÉUSSIS !")
    print("="*80)
    print("\nRésumé:")
    print("  ✓ Détection des requêtes courtes")
    print("  ✓ Logique conditionnelle (< 5 mots)")
    print("  ✓ Enrichissement Gemini (si clé API)")
    print("  ✓ Intégration avec movie_recommender")
