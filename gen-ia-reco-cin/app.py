"""
Application Flask pour le système de recommandation de films
"""
from flask import Flask, render_template, request, jsonify
import sys
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from recommender.movie_recommender import recommend_movies, load_movie_index

app = Flask(__name__)

# Charger l'index des films au démarrage
print("Chargement de l'index des films...")
df, embeddings = load_movie_index()
print(f"✓ {len(df)} films chargés avec succès")

@app.route('/')
def index():
    """Page d'accueil avec le formulaire"""
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    """Endpoint pour obtenir des recommandations"""
    try:
        data = request.get_json()
        
        # Extraire les paramètres
        description = data.get('description', '')
        similar_title = data.get('similar_title', '')
        action_intensity = data.get('action_intensity', 3)
        narrative_complexity = data.get('narrative_complexity', 3)
        darkness = data.get('darkness', 3)
        realism = data.get('realism', 3)
        period = data.get('period', None)
        
        # Afficher dans le terminal
        print("\n" + "="*80)
        print("🎬 NOUVELLE REQUÊTE DE RECOMMANDATION")
        print("="*80)
        print(f"📝 Description (Entrée) : {description if description else '[Vide]'}")
        print(f"🎬 Titre similaire      : {similar_title if similar_title else '[Vide]'}")
        print(f"📊 Action Intensity     : {action_intensity}/5")
        print(f"📊 Narrative Complexity : {narrative_complexity}/5")
        print(f"📊 Darkness             : {darkness}/5")
        print(f"📊 Realism              : {realism}/5")
        print(f"📅 Période              : {period if period else '[Toutes]'}")
        
        # Obtenir les recommandations (EF4.1 activé par défaut)
        recommendations = recommend_movies(
            description=description if description else None,
            similar_title=similar_title if similar_title else None,
            action_intensity=action_intensity,
            narrative_complexity=narrative_complexity,
            darkness=darkness,
            realism=realism,
            period=period,
            top_k=5,
            use_weights=True,
            enable_augmentation=True  # EF4.1 activé
        )
        
        print("\n🎯 RECOMMANDATIONS :")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec['titre']} ({rec['année']}) - Score: {rec['score']:.1f}%")
        print("="*80 + "\n")
        
        return jsonify({
            'success': True,
            'recommendations': recommendations
        })
        
    except Exception as e:
        print(f"❌ Erreur lors de la recommandation: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
