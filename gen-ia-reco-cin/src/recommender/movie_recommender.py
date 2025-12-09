"""
Module de recommandation de films et TV shows spécifiques basé sur les critères utilisateur.
Prend en charge :
- Description libre
- Titre similaire (preuve contextuelle)
- Échelle de Likert (intensité, complexité, noirceur, réalisme)
- Filtrage par période
"""
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import pickle
import pandas as pd

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
    
    # Intensité de l'action
    if action_intensity <= 2:
        query_parts.append("calme contemplatif lent")
    elif action_intensity >= 4:
        query_parts.append("action intense explosif dynamique")
    
    # Complexité narrative
    if narrative_complexity <= 2:
        query_parts.append("scénario simple linéaire direct")
    elif narrative_complexity >= 4:
        query_parts.append("scénario complexe puzzle mental intrigue")
    
    # Noirceur/Violence
    if darkness <= 2:
        query_parts.append("familial léger optimiste")
    elif darkness >= 4:
        query_parts.append("sombre violent mature dur")
    
    # Réalisme
    if realism <= 2:
        query_parts.append("réaliste documentaire historique")
    elif realism >= 4:
        query_parts.append("fantastique surréaliste imaginaire science-fiction")
    
    return " ".join(query_parts)

def calculate_likert_weights(
    action_intensity: int = 3,
    narrative_complexity: int = 3,
    darkness: int = 3,
    realism: int = 3
) -> Dict[str, float]:
    """
    Calcule des pondérations basées sur les scores Likert.
    Favorise les catégories correspondant aux préférences.
    """
    weights = {}
    
    # Action intensity
    if action_intensity >= 4:
        weights.update({"Action": 1.5, "Adventure": 1.3, "Thriller": 1.2})
    elif action_intensity <= 2:
        weights.update({"Drama": 1.3, "Documentary": 1.2, "Romance": 1.2})
    
    # Narrative complexity
    if narrative_complexity >= 4:
        weights.update({"Mystery": 1.4, "Thriller": 1.3, "Sci-Fi & Fantasy": 1.2})
    elif narrative_complexity <= 2:
        weights.update({"Comedy": 1.2, "Family": 1.1, "Animation": 1.1})
    
    # Darkness
    if darkness >= 4:
        weights.update({"Horror": 1.5, "Crime": 1.4, "War": 1.3, "Thriller": 1.2})
    elif darkness <= 2:
        weights.update({"Family": 1.4, "Comedy": 1.3, "Animation": 1.3, "Romance": 1.2})
    
    # Realism
    if realism >= 4:
        weights.update({"Sci-Fi & Fantasy": 1.5, "Fantasy": 1.4, "Animation": 1.2})
    elif realism <= 2:
        weights.update({"Documentary": 1.5, "Drama": 1.2, "History": 1.3, "War": 1.2})
    
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
    use_weights: bool = True
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
    
    Returns:
        Liste de dictionnaires avec les recommandations
    """
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
    
    # Construire la requête enrichie
    query = build_query_from_criteria(
        description, similar_title,
        action_intensity, narrative_complexity,
        darkness, realism
    )
    
    print(f"Requête construite : {query}")
    
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
        
        # Appliquer les poids aux catégories
        weighted_sims = []
        for i, (idx, row) in enumerate(df_filtered.iterrows()):
            category = row.get('Catégorie', '')
            weight = weights.get(category, 1.0)
            weighted_sims.append(sims[i] * weight)
        sims = np.array(weighted_sims)
    
    # Trier et récupérer top_k
    top_indices = np.argsort(sims)[::-1][:top_k]
    
    # Construire les résultats
    recommendations = []
    for idx in top_indices:
        row = df_filtered.iloc[idx]
        recommendations.append({
            "titre": row.get('Film', ''),
            "année": row.get('Année', ''),
            "catégorie": row.get('Catégorie', ''),
            "genre": row.get('Genre', ''),
            "blockid": row.get('BlockID', ''),
            "description": row.get('Description narrative', ''),
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
