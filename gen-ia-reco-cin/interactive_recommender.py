"""
Interface utilisateur interactive pour le système de recommandation de films/TV shows.
Permet à l'utilisateur de saisir ses préférences via un questionnaire structuré.
"""
import sys
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from recommender.movie_recommender import recommend_movies, load_movie_index

def print_header():
    """Affiche l'en-tête du programme."""
    print("\n" + "=" * 80)
    print(" " * 20 + "🎬 SYSTÈME DE RECOMMANDATION DE FILMS 🎬")
    print("=" * 80 + "\n")

def get_text_input(prompt: str, required: bool = False) -> str:
    """Obtient une entrée texte de l'utilisateur."""
    while True:
        value = input(prompt).strip()
        if value or not required:
            return value
        print("⚠️  Ce champ est obligatoire. Veuillez entrer une valeur.\n")

def get_likert_input(dimension: str, low_label: str, high_label: str) -> int:
    """Obtient un score Likert (1-5) de l'utilisateur."""
    while True:
        print(f"\n{dimension}")
        print(f"  1 = {low_label}")
        print(f"  5 = {high_label}")
        try:
            value = input("Votre choix (1-5) : ").strip()
            score = int(value)
            if 1 <= score <= 5:
                return score
            print("⚠️  Veuillez entrer un nombre entre 1 et 5.\n")
        except ValueError:
            print("⚠️  Veuillez entrer un nombre valide.\n")

def get_period_input() -> str:
    """Obtient la période temporelle souhaitée."""
    print("\n📅 Sur quelle période souhaitez-vous un film/série ?")
    print("  1. Présent - 2020 (Films récents)")
    print("  2. 2020 - 2015")
    print("  3. 2015 - 2010")
    print("  4. 2010 - 2000")
    print("  5. 2000 - 1980")
    print("  6. Avant 1980 (Classiques)")
    print("  7. Toutes périodes (aucun filtre)")
    
    period_map = {
        "1": "present-2020",
        "2": "2020-2015",
        "3": "2015-2010",
        "4": "2010-2000",
        "5": "2000-1980",
        "6": "1980<",
        "7": None
    }
    
    while True:
        choice = input("\nVotre choix (1-7) : ").strip()
        if choice in period_map:
            return period_map[choice]
        print("⚠️  Veuillez entrer un nombre entre 1 et 7.\n")

def collect_user_preferences():
    """Collecte toutes les préférences de l'utilisateur."""
    print_header()
    
    print("Répondez aux questions suivantes pour obtenir des recommandations personnalisées.\n")
    
    # 1. Description libre
    print("=" * 80)
    print("📝 DESCRIPTION LIBRE")
    print("=" * 80)
    description = get_text_input(
        "\nDécrivez en quelques phrases le type de film que vous avez envie de regarder maintenant :\n> "
    )
    
    # 2. Preuve contextuelle
    print("\n" + "=" * 80)
    print("🎬 PREUVE CONTEXTUELLE")
    print("=" * 80)
    similar_title = get_text_input(
        "\nDonnez un titre de film similaire (optionnel, appuyez sur Entrée pour passer) :\n> "
    )
    
    # 3. Échelle de Likert
    print("\n" + "=" * 80)
    print("📊 ÉCHELLE DE LIKERT")
    print("=" * 80)
    print("\nSur une échelle de 1 à 5, évaluez votre intérêt actuel pour les éléments suivants :")
    
    action_intensity = get_likert_input(
        "⚡ Intensité de l'Action",
        "Très calme/Contemplatif",
        "Action pure/Explosif"
    )
    
    narrative_complexity = get_likert_input(
        "🧩 Complexité Narrative",
        "Scénario linéaire/Simple",
        "Puzzle mental/Complexe"
    )
    
    darkness = get_likert_input(
        "🌑 Noirceur / Violence",
        "Familial/Léger",
        "Sombre/Violent"
    )
    
    realism = get_likert_input(
        "🌍 Réalisme",
        "Documentaire/Ancré dans le réel",
        "Fantastique/Surréaliste"
    )
    
    # 4. Période
    print("\n" + "=" * 80)
    print("📅 PÉRIODE")
    print("=" * 80)
    period = get_period_input()
    
    return {
        "description": description,
        "similar_title": similar_title,
        "action_intensity": action_intensity,
        "narrative_complexity": narrative_complexity,
        "darkness": darkness,
        "realism": realism,
        "period": period
    }

def display_recommendations(recommendations):
    """Affiche les recommandations de manière attractive."""
    print("\n" + "=" * 80)
    print("🎯 VOS RECOMMANDATIONS PERSONNALISÉES")
    print("=" * 80 + "\n")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. 🎬 {rec['titre']} ({rec['année']})")
        print(f"   📁 Type      : {rec['catégorie']}")
        print(f"   🎭 Genre     : {rec['genre']}")
        print(f"   ⭐ Score     : {rec['score']:.4f}")
        print(f"   📝 Synopsis  : {rec['description'][:150]}...")
        print()

def main():
    """Fonction principale du programme interactif."""
    try:
        # Charger l'index au démarrage
        print("\n⏳ Chargement de la base de données de films...")
        df, emb = load_movie_index()
        print(f"✓ Base de données chargée : {len(df)} films/séries disponibles")
        
        while True:
            # Collecter les préférences
            preferences = collect_user_preferences()
            
            # Afficher un résumé
            print("\n" + "=" * 80)
            print("📋 RÉSUMÉ DE VOTRE RECHERCHE")
            print("=" * 80)
            print(f"\n📝 Description        : {preferences['description'][:60]}...")
            if preferences['similar_title']:
                print(f"🎬 Similaire à        : {preferences['similar_title']}")
            print(f"⚡ Action             : {preferences['action_intensity']}/5")
            print(f"🧩 Complexité         : {preferences['narrative_complexity']}/5")
            print(f"🌑 Noirceur           : {preferences['darkness']}/5")
            print(f"🌍 Réalisme           : {preferences['realism']}/5")
            period_labels = {
                "present-2020": "Présent - 2020",
                "2020-2015": "2020 - 2015",
                "2015-2010": "2015 - 2010",
                "2010-2000": "2010 - 2000",
                "2000-1980": "2000 - 1980",
                "1980<": "Avant 1980",
                None: "Toutes périodes"
            }
            print(f"📅 Période            : {period_labels.get(preferences['period'], 'Toutes périodes')}")
            
            # Générer les recommandations
            print("\n⏳ Génération des recommandations...")
            recommendations = recommend_movies(
                description=preferences['description'],
                similar_title=preferences['similar_title'],
                action_intensity=preferences['action_intensity'],
                narrative_complexity=preferences['narrative_complexity'],
                darkness=preferences['darkness'],
                realism=preferences['realism'],
                period=preferences['period'],
                top_k=5,
                use_weights=True
            )
            
            # Afficher les recommandations
            display_recommendations(recommendations)
            
            # Demander si l'utilisateur veut continuer
            print("=" * 80)
            choice = input("\n🔄 Voulez-vous une nouvelle recherche ? (o/n) : ").strip().lower()
            if choice != 'o':
                print("\n👋 Merci d'avoir utilisé le système de recommandation ! À bientôt !\n")
                break
    
    except KeyboardInterrupt:
        print("\n\n👋 Programme interrompu. À bientôt !\n")
    except Exception as e:
        print(f"\n❌ Une erreur s'est produite : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
