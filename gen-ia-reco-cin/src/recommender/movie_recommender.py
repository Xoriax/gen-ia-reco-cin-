"""
Module de recommandation de films et TV shows spécifiques basé sur les critères utilisateur.
Prend en charge :
- Description libre
- Titre similaire (preuve contextuelle)
- Échelle de Likert (intensité, complexité, noirceur, réalisme)
- Filtrage par période
- EF4.1: Enrichissement automatique des requêtes courtes via expansion de contexte
"""
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import pickle
import pandas as pd
import sys

# Import du module d'augmentation (EF4.1)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from query_augmentation import augment_query_with_gemini

MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
_MODEL: Optional[SentenceTransformer] = None
DEFAULT_CSV = Path(__file__).resolve().parents[1] / "data" / "movies.csv"
DEFAULT_PICKLE = Path(__file__).resolve().parents[1] / "data" / "referentiel_movies.pkl"

def get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL

def load_movies_df(csv_path: str = None) -> pd.DataFrame:
    """Charge le DataFrame complet des films/TV shows."""
    p = Path(csv_path) if csv_path else DEFAULT_CSV
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    # Convertir l'année en numérique pour le filtrage
    df['Année'] = pd.to_numeric(df['Année'], errors='coerce')
    return df

def build_movie_embeddings(df: pd.DataFrame) -> np.ndarray:
    """
    Construit les embeddings pour chaque film/TV show.
    Combine : Titre + Description + Genre + Catégorie pour un embedding riche.
    """
    model = get_model()
    
    # Créer des textes enrichis pour chaque film
    texts = []
    for _, row in df.iterrows():
        # Combiner plusieurs champs pour un embedding plus riche
        text_parts = [
            f"Titre: {row.get('Film', '')}",
            f"Description: {row.get('Description narrative', '')}",
            f"Genre: {row.get('Genre', '')}",
            f"Catégorie: {row.get('Catégorie', '')}",
            f"BlockID: {row.get('BlockID', '')}"
        ]
        combined_text = " ".join(text_parts)
        texts.append(combined_text)
    
    return model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

def build_and_save_movie_index(csv_path: str = None, out_path: str = None) -> Tuple[pd.DataFrame, np.ndarray]:
    """Construit l'index de films avec embeddings et le sauvegarde."""
    df = load_movies_df(csv_path)
    embeddings = build_movie_embeddings(df)
    
    if out_path is None:
        out_path = str(DEFAULT_PICKLE)
    
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump({"model": MODEL_NAME, "df": df, "embeddings": embeddings}, f)
    
    return df, embeddings

def load_movie_index(path: str = None) -> Tuple[pd.DataFrame, np.ndarray]:
    """Charge l'index de films avec embeddings. Si absent, le construit."""
    p = Path(path) if path else DEFAULT_PICKLE
    if not p.exists():
        return build_and_save_movie_index()
    with p.open("rb") as f:
        data = pickle.load(f)
    return data["df"], data["embeddings"]

def embed_text(texts: List[str]) -> np.ndarray:
    """Encode une ou plusieurs chaînes."""
    model = get_model()
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

def filter_by_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    Filtre les films par période.
    Périodes supportées :
    - "present-2020" : >= 2020
    - "2020-2015" : 2015-2019
    - "2015-2010" : 2010-2014
    - "2010-2000" : 2000-2009
    - "2000-1980" : 1980-1999
    - "1980<" : < 1980
    """
    if period == "present-2020":
        return df[df['Année'] >= 2020]
    elif period == "2020-2015":
        return df[(df['Année'] >= 2015) & (df['Année'] < 2020)]
    elif period == "2015-2010":
        return df[(df['Année'] >= 2010) & (df['Année'] < 2015)]
    elif period == "2010-2000":
        return df[(df['Année'] >= 2000) & (df['Année'] < 2010)]
    elif period == "2000-1980":
        return df[(df['Année'] >= 1980) & (df['Année'] < 2000)]
    elif period == "1980<":
        return df[df['Année'] < 1980]
    else:
        return df  # Pas de filtre

def build_query_from_criteria(
    description: str = "",
    similar_title: str = "",
    action_intensity: int = 3,
    narrative_complexity: int = 3,
    darkness: int = 3,
    realism: int = 3
) -> str:
    """
    Construit une requête enrichie à partir des critères utilisateur.
    
    Args:
        description: Description libre du film recherché
        similar_title: Titre d'un film similaire
        action_intensity: 1-5 (1=Calme, 5=Action pure)
        narrative_complexity: 1-5 (1=Simple, 5=Complexe)
        darkness: 1-5 (1=Familial, 5=Sombre/Violent)
        realism: 1-5 (1=Documentaire, 5=Fantastique)
    
    Returns:
        Requête textuelle enrichie
    """
    query_parts = []
    
    # Description libre (priorité haute)
    if description:
        query_parts.append(description)
    
    # Titre similaire (preuve contextuelle)
    if similar_title:
        query_parts.append(f"similaire à {similar_title}")
    
    # Action intensity (English for consistency with movies.csv)
    if action_intensity <= 2:
        query_parts.append("calm contemplative slow peaceful")
    elif action_intensity >= 4:
        query_parts.append("action intense explosive dynamic fight combat")
    
    # Narrative complexity
    if narrative_complexity <= 2:
        query_parts.append("simple plot linear straightforward direct")
    elif narrative_complexity >= 4:
        query_parts.append("complex puzzle mind-bending intricate mystery")
    
    # Darkness/Violence
    if darkness <= 2:
        query_parts.append("family friendly lighthearted optimistic")
    elif darkness >= 4:
        query_parts.append("dark violent mature gritty psychological")
    
    # Realism vs Fantasy
    if realism <= 2:
        query_parts.append("realistic documentary historical true story")
    elif realism >= 4:
        query_parts.append("fantasy fantastical imaginative science fiction magical")
    
    return " ".join(query_parts)

def calculate_likert_weights(
    action_intensity: int = 3,
    narrative_complexity: int = 3,
    darkness: int = 3,
    realism: int = 3
) -> Dict[str, float]:
    """
    Calculates weights based on Likert scores.
    Maps user preferences to genre weights for scoring.
    Uses genres found in the movies.csv file.
    
    Args:
        action_intensity: 1-5 (1=Calm, 5=Action)
        narrative_complexity: 1-5 (1=Simple, 5=Complex)
        darkness: 1-5 (1=Family, 5=Dark/Violent)
        realism: 1-5 (1=Realistic, 5=Fantasy)
    
    Returns:
        Dictionary mapping genres to weight multipliers
    """
    weights = {}
    
    # Action intensity (genres with strong action)
    if action_intensity >= 4:
        weights.update({
            "Action": 1.5,
            "Adventure": 1.3,
            "Thriller": 1.2,
            "War": 1.2
        })
    elif action_intensity <= 2:
        weights.update({
            "Drama": 1.3,
            "Documentary": 1.2,
            "Romance": 1.2,
            "Comedy": 1.1
        })
    
    # Narrative complexity
    if narrative_complexity >= 4:
        weights.update({
            "Mystery": 1.4,
            "Thriller": 1.3,
            "Science Fiction": 1.2,
            "Drama": 1.2
        })
    elif narrative_complexity <= 2:
        weights.update({
            "Comedy": 1.2,
            "Family": 1.1,
            "Animation": 1.1,
            "Romance": 1.1
        })
    
    # Darkness (violent/dark genres)
    if darkness >= 4:
        weights.update({
            "Horror": 1.5,
            "Crime": 1.4,
            "War": 1.3,
            "Thriller": 1.2,
            "Drama": 1.1
        })
    elif darkness <= 2:
        weights.update({
            "Family": 1.4,
            "Comedy": 1.3,
            "Animation": 1.3,
            "Romance": 1.2
        })
    
    # Realism vs Fantasy
    if realism >= 4:
        weights.update({
            "Science Fiction": 1.5,
            "Fantasy": 1.4,
            "Animation": 1.2
        })
    elif realism <= 2:
        weights.update({
            "Documentary": 1.5,
            "Drama": 1.2,
            "History": 1.3,
            "War": 1.2
        })
    
    return weights

def recommend_movies(
    description: str = "",
    similar_title: str = "",
    action_intensity: int = 3,
    narrative_complexity: int = 3,
    darkness: int = 3,
    realism: int = 3,
    period: str = None,
    top_k: int = 3,
    use_weights: bool = True,
    enable_augmentation: bool = True
) -> List[Dict]:
    """
    Recommande des films/TV shows spécifiques basés sur les critères utilisateur.
    
    Args:
        description: Description libre du film recherché
        similar_title: Titre d'un film similaire
        action_intensity: 1-5 (Intensité de l'action)
        narrative_complexity: 1-5 (Complexité narrative)
        darkness: 1-5 (Noirceur/Violence)
        realism: 1-5 (Réalisme vs Fantastique)
        period: Période temporelle (ex: "present-2020", "2010-2000")
        top_k: Nombre de recommandations
        use_weights: Appliquer la pondération Likert
        enable_augmentation: EF4.1 - Activer l'enrichissement automatique des requêtes courtes
    
    Returns:
        Liste de dictionnaires avec les recommandations
    """
    # EF4.1 : Augmentation de la description si elle est trop courte
    if description and enable_augmentation:
        description = augment_query_with_gemini(description)
    
    # Charger l'index
    df, embeddings = load_movie_index()
    
    # Filtrer par période si spécifié
    if period:
        df_filtered = filter_by_period(df, period)
        if len(df_filtered) == 0:
            print(f"Aucun film trouvé pour la période {period}")
            df_filtered = df
        # Recalculer les indices
        filtered_indices = df_filtered.index.tolist()
        embeddings_filtered = embeddings[filtered_indices]
    else:
        df_filtered = df
        embeddings_filtered = embeddings
    
    # Build enriched query
    query = build_query_from_criteria(
        description, similar_title,
        action_intensity, narrative_complexity,
        darkness, realism
    )
    
    print(f"[Query] Built query: {query}")
    
    # Encoder la requête
    q_emb = embed_text([query])
    
    # Calculer similarités
    sims = cosine_similarity(q_emb, embeddings_filtered)[0]
    
    # Appliquer pondérations si demandé
    if use_weights:
        weights = calculate_likert_weights(
            action_intensity, narrative_complexity,
            darkness, realism
        )
        
        # Apply weights to genres (not category)
        # Each movie can have multiple genres separated by commas
        weighted_sims = []
        for i, (idx, row) in enumerate(df_filtered.iterrows()):
            genre_str = str(row.get('Genre', ''))
            genres_list = [g.strip() for g in genre_str.split(',')]
            
            # Find maximum weight among all genres of this movie
            max_weight = 1.0
            for genre in genres_list:
                if genre in weights:
                    max_weight = max(max_weight, weights[genre])
            
            weighted_sims.append(sims[i] * max_weight)
        sims = np.array(weighted_sims)
    
    # Sort and get top_k
    top_indices = np.argsort(sims)[::-1][:top_k]
    
    # Build results
    recommendations = []
    for idx in top_indices:
        row = df_filtered.iloc[idx]
        recommendations.append({
            "titre": str(row.get('Film', '')),
            "année": int(row.get('Année', 0)),
            "catégorie": str(row.get('Catégorie', '')),
            "genre": str(row.get('Genre', '')),
            "blockid": str(row.get('BlockID', '')),
            "description": str(row.get('Description narrative', '')),
            "score": float(sims[idx])
        })
    
    return recommendations

if __name__ == "__main__":
    print("Construction de l'index des films...")
    df, emb = load_movie_index()
    print(f"Index construit : {len(df)} films/shows")
    
    # Test de recommandation
    print("\n=== Test de recommandation ===")
    recs = recommend_movies(
        description="un film dramatique intense avec du suspense",
        action_intensity=4,
        narrative_complexity=4,
        darkness=4,
        realism=2,
        period="2010-2000",
        top_k=5
    )
    
    print("\nTop 5 recommandations :")
    for i, rec in enumerate(recs, 1):
        print(f"\n{i}. {rec['titre']} ({rec['année']})")
        print(f"   Catégorie: {rec['catégorie']} | Genre: {rec['genre']}")
        print(f"   Score: {rec['score']:.4f}")
        print(f"   Description: {rec['description'][:100]}...")
