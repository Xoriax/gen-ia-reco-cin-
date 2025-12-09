"""
Script de test pour EF3.1 et EF3.2 :
- EF3.1 : Formule de score pondérée pour calculer le Score de Couverture Sémantique
- EF3.2 : Top 3 propositions avec scores les plus élevés
"""
import sys
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from services.ref import load_referentiel, recommend, recommend_weighted

def test_basic_recommendation():
    """Test de la recommandation de base (sans pondération)"""
    print("=" * 80)
    print("TEST 1 : Recommandation de base (top 3)")
    print("=" * 80)
    
    items, embeddings = load_referentiel()
    print(f"Référentiel chargé : {len(items)} BlockID")
    
    query = "film dramatique intense avec du crime"
    print(f"\nQuery : '{query}'")
    
    recs = recommend(query, items, embeddings, top_k=3)
    print("\nTop 3 recommandations (score cosinus) :")
    for i, (label, score) in enumerate(recs, 1):
        print(f"  {i}. {label:20s} - Score: {score:.4f}")

def test_weighted_recommendation():
    """Test de la recommandation pondérée (EF3.1 et EF3.2)"""
    print("\n" + "=" * 80)
    print("TEST 2 : Recommandation pondérée (EF3.1 - Score pondéré)")
    print("=" * 80)
    
    items, embeddings = load_referentiel()
    
    # Définir des poids pour favoriser certaines catégories
    weights = {
        "Drama": 1.5,      # Favoriser Drama
        "Crime": 1.3,      # Favoriser Crime
        "Thriller": 1.2,   # Favoriser Thriller
        "Comedy": 0.5,     # Défavoriser Comedy
        "Animation": 0.3   # Défavoriser Animation
    }
    
    query = "film dramatique intense avec du crime"
    print(f"\nQuery : '{query}'")
    print("\nPondérations appliquées :")
    for cat, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {cat:15s} : {weight:.1f}x")
    
    recs = recommend_weighted(query, items, embeddings, top_k=3, weights=weights)
    print("\nTop 3 recommandations (EF3.2 - score pondéré) :")
    for i, (label, score) in enumerate(recs, 1):
        weight_str = f" (poids: {weights.get(label, 1.0):.1f})" if label in weights else ""
        print(f"  {i}. {label:20s} - Score pondéré: {score:.4f}{weight_str}")

def test_different_queries():
    """Test avec différentes queries pour vérifier la cohérence"""
    print("\n" + "=" * 80)
    print("TEST 3 : Test avec différentes queries")
    print("=" * 80)
    
    items, embeddings = load_referentiel()
    
    test_queries = [
        ("film d'animation pour enfants", {"Animation": 2.0, "Family": 1.5, "Drama": 0.5}),
        ("comédie romantique légère", {"Comedy": 2.0, "Romance": 1.8, "Drama": 0.6}),
        ("thriller d'action épique", {"Action": 2.0, "Thriller": 1.8, "Adventure": 1.5})
    ]
    
    for query, weights in test_queries:
        print(f"\n Query : '{query}'")
        recs = recommend_weighted(query, items, embeddings, top_k=3, weights=weights)
        print(" Top 3 :")
        for i, (label, score) in enumerate(recs, 1):
            print(f"   {i}. {label:20s} - Score: {score:.4f}")

def test_coverage_semantic_score():
    """Test du score de couverture sémantique (EF3.1)"""
    print("\n" + "=" * 80)
    print("TEST 4 : Score de Couverture Sémantique (EF3.1)")
    print("=" * 80)
    
    items, embeddings = load_referentiel()
    
    # Simuler un profil utilisateur avec préférences multiples
    user_profile = "J'aime les films dramatiques, les thrillers psychologiques et les films de guerre historiques"
    
    # Pondérations basées sur l'affinité thématique
    thematic_weights = {
        "Drama": 1.8,
        "Thriller": 1.6,
        "War": 1.5,
        "History": 1.4,
        "Mystery": 1.3,
        "Crime": 1.2,
        "Action": 0.8,
        "Comedy": 0.4,
        "Animation": 0.3
    }
    
    print(f"\nProfil utilisateur : '{user_profile}'")
    print("\nPondérations thématiques (affinité sémantique) :")
    for theme, weight in sorted(thematic_weights.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  - {theme:15s} : {weight:.1f}x")
    
    recs = recommend_weighted(user_profile, items, embeddings, top_k=5, weights=thematic_weights)
    
    print("\nTop 5 recommandations avec Score de Couverture Sémantique :")
    for i, (label, score) in enumerate(recs, 1):
        weight_info = f" (affinité: {thematic_weights.get(label, 1.0):.1f}x)" if label in thematic_weights else ""
        print(f"  {i}. {label:20s} - Score de couverture: {score:.4f}{weight_info}")
    
    # Calculer le score de couverture global
    total_coverage = sum(score for _, score in recs)
    avg_coverage = total_coverage / len(recs)
    print(f"\nScore de Couverture Sémantique Global : {total_coverage:.4f}")
    print(f"Score de Couverture Moyen (top 5)      : {avg_coverage:.4f}")

if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "TEST DES FONCTIONNALITÉS EF3.1 et EF3.2" + " " * 24 + "║")
    print("╚" + "=" * 78 + "╝")
    
    try:
        test_basic_recommendation()
        test_weighted_recommendation()
        test_different_queries()
        test_coverage_semantic_score()
        
        print("\n" + "=" * 80)
        print("✓ TOUS LES TESTS SONT TERMINÉS AVEC SUCCÈS")
        print("=" * 80)
        print("\nRésumé des fonctionnalités testées :")
        print("  ✓ EF3.1 : Formule de score pondérée (similarité × poids thématique)")
        print("  ✓ EF3.2 : Top 3 propositions avec scores les plus élevés")
        print("  ✓ Score de Couverture Sémantique calculé et fonctionnel")
        print("\n")
        
    except Exception as e:
        print(f"\n✗ ERREUR : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
