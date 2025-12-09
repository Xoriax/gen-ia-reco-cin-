"""
Script de test interactif pour le système de recommandation de films/TV shows.
Teste toutes les fonctionnalités demandées :
- Description libre
- Titre similaire (preuve contextuelle)
- Échelle de Likert
- Filtrage par période
"""
import sys
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from recommender.movie_recommender import (
    recommend_movies,
    load_movie_index,
    build_query_from_criteria,
    calculate_likert_weights
)

def print_separator(char="=", length=80):
    print(char * length)

def print_header(text):
    print_separator()
    print(f"  {text}")
    print_separator()

def display_recommendations(recommendations, title="Recommandations"):
    """Affiche les recommandations de manière formatée."""
    print(f"\n{title} :\n")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec['titre']} ({rec['année']})")
        print(f"     📁 Catégorie: {rec['catégorie']}")
        print(f"     🎭 Genre: {rec['genre']}")
        print(f"     ⭐ Score: {rec['score']:.4f}")
        print(f"     📝 {rec['description'][:120]}...")
        print()

def test_scenario_1():
    """Test avec description libre uniquement."""
    print_header("TEST 1 : Description Libre")
    
    description = "Je veux un film dramatique intense avec du crime et du suspense"
    
    print(f"\n📝 Description : \"{description}\"\n")
    
    recs = recommend_movies(
        description=description,
        top_k=5,
        use_weights=False
    )
    
    display_recommendations(recs, "Top 5 Films")

def test_scenario_2():
    """Test avec titre similaire (preuve contextuelle)."""
    print_header("TEST 2 : Preuve Contextuelle (Titre Similaire)")
    
    similar_title = "The Shawshank Redemption"
    
    print(f"\n🎬 Titre similaire : \"{similar_title}\"\n")
    
    recs = recommend_movies(
        similar_title=similar_title,
        top_k=5,
        use_weights=False
    )
    
    display_recommendations(recs, "Films similaires")

def test_scenario_3():
    """Test avec échelle de Likert."""
    print_header("TEST 3 : Échelle de Likert")
    
    print("\n📊 Critères Likert (1-5) :")
    print("  • Intensité de l'Action    : 5 (Action pure/Explosif)")
    print("  • Complexité Narrative     : 4 (Complexe)")
    print("  • Noirceur/Violence        : 4 (Sombre/Violent)")
    print("  • Réalisme                 : 2 (Ancré dans le réel)")
    
    # Calculer les poids appliqués
    weights = calculate_likert_weights(
        action_intensity=5,
        narrative_complexity=4,
        darkness=4,
        realism=2
    )
    
    print("\n⚖️  Pondérations thématiques calculées :")
    for category, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"     - {category:20s} : {weight:.1f}x")
    
    recs = recommend_movies(
        description="film intense avec beaucoup d'action",
        action_intensity=5,
        narrative_complexity=4,
        darkness=4,
        realism=2,
        top_k=5,
        use_weights=True
    )
    
    display_recommendations(recs, "Top 5 Films (avec pondération Likert)")

def test_scenario_4():
    """Test avec filtrage par période."""
    print_header("TEST 4 : Filtrage par Période")
    
    periods = [
        ("present-2020", "Films récents (≥ 2020)"),
        ("2010-2000", "Années 2000"),
        ("1980<", "Classiques (< 1980)")
    ]
    
    description = "comédie romantique légère"
    
    for period, period_label in periods:
        print(f"\n📅 Période : {period_label}")
        
        recs = recommend_movies(
            description=description,
            period=period,
            top_k=3,
            use_weights=False
        )
        
        if recs:
            print(f"\nTop 3 :")
            for i, rec in enumerate(recs, 1):
                print(f"  {i}. {rec['titre']} ({rec['année']}) - {rec['catégorie']}")

def test_scenario_5():
    """Test combiné : tous les critères ensemble."""
    print_header("TEST 5 : Scénario Complet (Tous les Critères)")
    
    print("\n📋 Critères de recherche :")
    print("  • Description      : \"film de science-fiction épique avec des voyages dans l'espace\"")
    print("  • Titre similaire  : \"Interstellar\"")
    print("  • Action           : 4 (Intense)")
    print("  • Complexité       : 5 (Très complexe)")
    print("  • Noirceur         : 3 (Modéré)")
    print("  • Réalisme         : 5 (Fantastique/Surréaliste)")
    print("  • Période          : 2015-2010")
    
    recs = recommend_movies(
        description="film de science-fiction épique avec des voyages dans l'espace",
        similar_title="Interstellar",
        action_intensity=4,
        narrative_complexity=5,
        darkness=3,
        realism=5,
        period="2015-2010",
        top_k=5,
        use_weights=True
    )
    
    display_recommendations(recs, "Top 5 Recommandations (Recherche Complète)")

def test_scenario_6():
    """Test avec profil utilisateur complexe."""
    print_header("TEST 6 : Profil Utilisateur Complexe")
    
    print("\n👤 Profil utilisateur :")
    print("  \"J'aime les thrillers psychologiques sombres avec des intrigues complexes")
    print("  et des rebondissements. J'ai adoré Se7en et Fight Club.\"")
    
    print("\n📊 Paramètres Likert :")
    print("  • Action           : 3 (Modéré)")
    print("  • Complexité       : 5 (Puzzle mental)")
    print("  • Noirceur         : 5 (Très sombre)")
    print("  • Réalisme         : 2 (Réaliste)")
    print("  • Période          : 2000-1980")
    
    recs = recommend_movies(
        description="thriller psychologique sombre avec intrigues complexes et rebondissements",
        similar_title="Se7en",
        action_intensity=3,
        narrative_complexity=5,
        darkness=5,
        realism=2,
        period="2000-1980",
        top_k=5,
        use_weights=True
    )
    
    display_recommendations(recs, "Recommandations pour ce profil")

def test_query_building():
    """Test de la construction de requête enrichie."""
    print_header("TEST 7 : Construction de Requête Enrichie")
    
    scenarios = [
        {
            "desc": "film d'action épique",
            "similar": "Mad Max",
            "action": 5, "complex": 2, "dark": 3, "real": 3
        },
        {
            "desc": "conte de fées magique",
            "similar": "Spirited Away",
            "action": 1, "complex": 3, "dark": 1, "real": 5
        },
        {
            "desc": "documentaire historique",
            "similar": "",
            "action": 1, "complex": 4, "dark": 3, "real": 1
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        query = build_query_from_criteria(
            description=scenario["desc"],
            similar_title=scenario["similar"],
            action_intensity=scenario["action"],
            narrative_complexity=scenario["complex"],
            darkness=scenario["dark"],
            realism=scenario["real"]
        )
        
        print(f"\nScénario {i} :")
        print(f"  Description   : {scenario['desc']}")
        print(f"  Similaire à   : {scenario['similar']}")
        print(f"  Likert        : Action={scenario['action']}, Complexité={scenario['complex']}, "
              f"Noirceur={scenario['dark']}, Réalisme={scenario['real']}")
        print(f"  ➡️  Requête enrichie :")
        print(f"     \"{query}\"")

if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 10 + "SYSTÈME DE RECOMMANDATION DE FILMS/TV SHOWS" + " " * 25 + "║")
    print("║" + " " * 15 + "Test des Fonctionnalités Complètes" + " " * 29 + "║")
    print("╚" + "=" * 78 + "╝")
    
    try:
        # Charger l'index une fois au début
        print("\n⏳ Chargement de l'index des films...")
        df, emb = load_movie_index()
        print(f"✓ Index chargé : {len(df)} films/TV shows disponibles\n")
        
        # Exécuter tous les scénarios de test
        test_scenario_1()
        test_scenario_2()
        test_scenario_3()
        test_scenario_4()
        test_scenario_5()
        test_scenario_6()
        test_query_building()
        
        print_separator()
        print("✅ TOUS LES TESTS SONT TERMINÉS AVEC SUCCÈS")
        print_separator()
        
        print("\n📊 Résumé des fonctionnalités testées :")
        print("  ✓ Description libre")
        print("  ✓ Titre similaire (preuve contextuelle)")
        print("  ✓ Échelle de Likert (4 dimensions)")
        print("  ✓ Filtrage par période (6 tranches temporelles)")
        print("  ✓ Pondération sémantique")
        print("  ✓ Recommandations de titres spécifiques (films/TV shows)")
        print("\n")
        
    except Exception as e:
        print(f"\n✗ ERREUR : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
