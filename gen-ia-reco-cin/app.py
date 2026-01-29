"""
Flask application for movie recommendation system
Integrates both web interface and interactive CLI into one application
"""
from flask import Flask, render_template, request, jsonify
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.recommender.movie_recommender import recommend_movies, load_movie_index

app = Flask(__name__)

# Load movie index on startup
print("Loading movie index...")
df, embeddings = load_movie_index()
print(f"✓ {len(df)} movies loaded successfully")

@app.route('/')
def index():
    """Home page with interactive questionnaire (Hybrid questionnaire - EF1.1)"""
    return render_template('interactive.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    """Endpoint to get movie recommendations"""
    try:
        data = request.get_json()
        
        # Extract parameters
        description = data.get('description', '')
        similar_title = data.get('similar_title', '')
        action_intensity = data.get('action_intensity', 3)
        narrative_complexity = data.get('narrative_complexity', 3)
        darkness = data.get('darkness', 3)
        realism = data.get('realism', 3)
        period = data.get('period', None)
        
        # Log to console
        print("\n" + "="*80)
        print("🎬 NEW RECOMMENDATION REQUEST")
        print("="*80)
        print(f"📝 Description (Input)    : {description if description else '[Empty]'}")
        print(f"🎬 Similar Title          : {similar_title if similar_title else '[Empty]'}")
        print(f"📊 Action Intensity       : {action_intensity}/5")
        print(f"📊 Narrative Complexity   : {narrative_complexity}/5")
        print(f"📊 Darkness               : {darkness}/5")
        print(f"📊 Realism                : {realism}/5")
        print(f"📅 Period                 : {period if period else '[All]'}")
        
        # Get recommendations (EF4.1 enabled by default)
        result = recommend_movies(
            description=description if description else None,
            similar_title=similar_title if similar_title else None,
            action_intensity=action_intensity,
            narrative_complexity=narrative_complexity,
            darkness=darkness,
            realism=realism,
            period=period,
            top_k=3,  # EF3.2: Top 3 recommendations
            use_weights=True,
            enable_augmentation=True  # EF4.1 enabled
        )
        
        # Extract recommendations and GenAI justification
        recommendations = result.get('recommendations', result) if isinstance(result, dict) and 'recommendations' in result else result
        genai_justification = result.get('genai_justification') if isinstance(result, dict) else None
        
        print("\n🎯 RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec['title']} ({rec['year']}) - Score: {rec['score']:.1%}")
        
        if genai_justification:
            print(f"\n💡 GenAI Justification: {genai_justification}")
        
        print("="*80 + "\n")
        
        return jsonify({
            'success': True,
            'genai_justification': genai_justification,
            'recommendations': recommendations
        })
        
    except Exception as e:
        print(f"❌ Error during recommendation: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
