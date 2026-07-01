"""
Movie recommendation module for films and TV shows based on user criteria.
Supports:
- Free-form description
- Similar title (contextual proof via semantic search)
- Likert scale (intensity, complexity, darkness, realism)
- Period filtering
- EF4.1: Automatic enrichment of short queries via context expansion
- Local semantic search: No external APIs needed (pure Hugging Face)
"""
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import pickle
import pandas as pd
import sys

# Import augmentation module (EF4.1)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.query_augmentation import augment_query_with_openrouter
from src.utils.semantic_search import find_similar_movies_in_dataset, aggregate_similar_movies_context
from src.utils.genai_justification import generate_recommendation_justification

MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
_MODEL: Optional[SentenceTransformer] = None
DEFAULT_CSV = Path(__file__).resolve().parents[1] / "data" / "movies.csv"
DEFAULT_PARQUET = Path(__file__).resolve().parents[1] / "data" / "movies.parquet"
DEFAULT_PICKLE = Path(__file__).resolve().parents[1] / "data" / "referentiel_movies.pkl"

# In-memory cache of the loaded index (df, full embeddings, description-only
# embeddings), so the pkl is read from disk at most once per process.
_INDEX_CACHE: Optional[dict] = None

def get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL

def load_movies_df(csv_path: str = None) -> pd.DataFrame:
    """
    Load complete DataFrame of movies/TV shows. Prefers the Parquet twin
    (faster to parse than CSV, especially at 10k+ rows) and only falls back
    to CSV if no explicit csv_path was given and the Parquet file is missing.
    """
    if csv_path is None and DEFAULT_PARQUET.exists():
        # Parquet was written from the same all-string dataframe as the CSV
        # (see tmdb_extract.py), so dtypes/empty-string handling match already.
        df = pd.read_parquet(DEFAULT_PARQUET)
    else:
        p = Path(csv_path) if csv_path else DEFAULT_CSV
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
    # Convert year to numeric for filtering
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    return df

def build_movie_embeddings(df: pd.DataFrame) -> np.ndarray:
    """
    Build embeddings for each movie/TV show.
    Combines: Title + Description + Genre + Category for rich embedding.
    """
    model = get_model()
    
    # Create enriched texts for each movie
    texts = []
    for _, row in df.iterrows():
        # Combine multiple fields for richer embedding
        text_parts = [
            f"Title: {row.get('Title', '')}",
            f"Description: {row.get('Description', '')}",
            f"Genre: {row.get('Genre', '')}",
            f"Category: {row.get('Category', '')}",
            f"BlockID: {row.get('BlockID', '')}"
        ]
        combined_text = " ".join(text_parts)
        texts.append(combined_text)
    
    return model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

def build_description_embeddings(df: pd.DataFrame) -> np.ndarray:
    """
    Pre-computes one embedding per movie description only (not the combined
    Title/Description/Genre/Category text used by build_movie_embeddings).
    Stored in the index so recommend_movies never has to re-encode the whole
    dataset's descriptions on every request.
    """
    model = get_model()
    descriptions = df['Description'].fillna('').astype(str).tolist()
    return model.encode(descriptions, show_progress_bar=True, convert_to_numpy=True)

def build_and_save_movie_index(csv_path: str = None, out_path: str = None) -> Tuple[pd.DataFrame, np.ndarray]:
    """Build movie index (full + description embeddings) and save it."""
    global _INDEX_CACHE
    df = load_movies_df(csv_path)
    embeddings = build_movie_embeddings(df)
    desc_embeddings = build_description_embeddings(df)

    if out_path is None:
        out_path = str(DEFAULT_PICKLE)

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump({
            "model": MODEL_NAME,
            "df": df,
            "embeddings": embeddings,
            "desc_embeddings": desc_embeddings,
        }, f)

    _INDEX_CACHE = {"df": df, "embeddings": embeddings, "desc_embeddings": desc_embeddings}
    return df, embeddings

def _load_index_data(path: str = None) -> dict:
    """
    Loads (and caches in-process) the full index dict: df, combined
    embeddings, and description-only embeddings. Rebuilds the description
    embeddings once (and re-saves the pkl) if loading an older index that
    predates this field, so existing pkls upgrade transparently.
    """
    global _INDEX_CACHE
    if _INDEX_CACHE is not None:
        return _INDEX_CACHE

    p = Path(path) if path else DEFAULT_PICKLE
    if not p.exists():
        build_and_save_movie_index(out_path=path)
        return _INDEX_CACHE

    with p.open("rb") as f:
        data = pickle.load(f)

    desc_embeddings = data.get("desc_embeddings")
    if desc_embeddings is None:
        desc_embeddings = build_description_embeddings(data["df"])
        data["desc_embeddings"] = desc_embeddings
        with p.open("wb") as f:
            pickle.dump(data, f)

    _INDEX_CACHE = {"df": data["df"], "embeddings": data["embeddings"], "desc_embeddings": desc_embeddings}
    return _INDEX_CACHE

def load_movie_index(path: str = None) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load movie index with embeddings. If missing, build it."""
    data = _load_index_data(path)
    return data["df"], data["embeddings"]

def embed_text(texts: List[str]) -> np.ndarray:
    """Encode one or more strings."""
    model = get_model()
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

def filter_by_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    Filter movies by period.
    Supported periods:
    - "present-2020" : >= 2020
    - "2020-2015" : 2015-2019
    - "2015-2010" : 2010-2014
    - "2010-2000" : 2000-2009
    - "2000-1980" : 1980-1999
    - "1980<" : < 1980
    """
    if period == "present-2020":
        return df[df['Year'] >= 2020]
    elif period == "2020-2015":
        return df[(df['Year'] >= 2015) & (df['Year'] < 2020)]
    elif period == "2015-2010":
        return df[(df['Year'] >= 2010) & (df['Year'] < 2015)]
    elif period == "2010-2000":
        return df[(df['Year'] >= 2000) & (df['Year'] < 2010)]
    elif period == "2000-1980":
        return df[(df['Year'] >= 1980) & (df['Year'] < 2000)]
    elif period == "1980<":
        return df[df['Year'] < 1980]
    else:
        return df  # No filter

def build_query_from_criteria(
    description: str = "",
    similar_title: str = "",
    action_intensity: int = 3,
    narrative_complexity: int = 3,
    darkness: int = 3,
    realism: int = 3
) -> str:
    """
    Build enriched query from user criteria.
    
    Args:
        description: Free-form description of desired movie
        similar_title: Title of a similar movie
        action_intensity: 1-5 (1=Calm, 5=Action)
        narrative_complexity: 1-5 (1=Simple, 5=Complex)
        darkness: 1-5 (1=Family, 5=Dark/Violent)
        realism: 1-5 (1=Realistic, 5=Fantasy)
    
    Returns:
        Enriched textual query
    """
    query_parts = []
    
    # Free-form description (high priority)
    if description:
        query_parts.append(description)
    
    # Similar title (contextual proof)
    if similar_title:
        query_parts.append(f"similar to {similar_title}")
    
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
    Calculate weights based on Likert scores.
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

def process_similar_title(similar_title: str, df: pd.DataFrame, embeddings: np.ndarray) -> Optional[str]:
    """
    Process similar title using local semantic search.
    Finds similar movies in our dataset and aggregates their context.
    No external APIs - pure Hugging Face embeddings!
    
    Args:
        similar_title: Movie title provided by user
        df: Movies dataframe
        embeddings: Pre-computed embeddings for all movies
        
    Returns:
        Aggregated context from similar movies or None
    """
    if not similar_title:
        return None
    
    print(f"\n[SIMILAR TITLE] Processing: '{similar_title}'")
    
    # Find similar movies in our dataset
    similar_movies = find_similar_movies_in_dataset(
        similar_title,
        df,
        embeddings,
        top_k=5
    )
    
    if not similar_movies:
        print(f"   Warning: no similar movies found in dataset")
        return None

    # Aggregate descriptions from found similar movies
    aggregated_context = aggregate_similar_movies_context(similar_movies, max_descriptions=3)
    print(f"   OK: aggregated context from {len(similar_movies)} similar movies")
    
    return aggregated_context

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
    enable_augmentation: bool = True,
    enable_justification: bool = True
) -> List[Dict]:
    """
    Recommend specific movies/TV shows based on user criteria.

    Uses weighted scoring formula:
    SCORE_FINAL = (FREE_FORM × 0.25) + (SIMILAR_TITLE × 0.25) + (GENRE_WEIGHT × 0.20) + (COSINE × 0.30)

    All similarity calculations use Hugging Face SentenceTransformer embeddings (no external APIs).

    Args:
        description: Free-form description of desired movie
        similar_title: Title of a similar movie (matched semantically in dataset)
        action_intensity: 1-5 (Action intensity)
        narrative_complexity: 1-5 (Narrative complexity)
        darkness: 1-5 (Darkness/Violence)
        realism: 1-5 (Realism vs Fantasy)
        period: Time period (ex: "present-2020", "2010-2000")
        top_k: Number of recommendations
        use_weights: Apply Likert weighting
        enable_augmentation: EF4.1 - Enable automatic enrichment of short queries
        enable_justification: EF4.3 - Generate the GenAI justification text.
            Set to False for pure scoring comparisons (e.g. the evaluation
            dashboard's before/after test) to skip the LLM call entirely.

    Returns:
        List of dictionaries with recommendations
    """
    # EF4.1: Augment description if too short
    if description and enable_augmentation:
        description = augment_query_with_openrouter(description)
    
    # Load index (df, combined embeddings, precomputed description embeddings)
    index_data = _load_index_data()
    df = index_data["df"]
    embeddings = index_data["embeddings"]
    desc_embeddings = index_data["desc_embeddings"]

    # Filter by period if specified
    if period:
        df_filtered = filter_by_period(df, period)
        if len(df_filtered) == 0:
            print(f"No movies found for period {period}")
            df_filtered = df
        # Recalculate indices
        filtered_indices = df_filtered.index.tolist()
        embeddings_filtered = embeddings[filtered_indices]
        desc_embeddings_filtered = desc_embeddings[filtered_indices]
    else:
        df_filtered = df
        embeddings_filtered = embeddings
        desc_embeddings_filtered = desc_embeddings

    # Build enriched query
    query = build_query_from_criteria(
        description, similar_title,
        action_intensity, narrative_complexity,
        darkness, realism
    )

    print(f"[Query] Built query: {query}")

    # Encode full query
    q_emb = embed_text([query])

    # Calculate full similarities (Title + Description + Genre + Category)
    full_sims = cosine_similarity(q_emb, embeddings_filtered)[0]

    # Calculate description-specific similarity using the precomputed
    # description embeddings (no re-encoding of the dataset at request time)
    if description:
        desc_emb = embed_text([description])
        desc_sims = cosine_similarity(desc_emb, desc_embeddings_filtered)[0]
    else:
        # If no description, use uniform similarity
        desc_sims = np.full(len(df_filtered), 0.5)
    
    # Process similar title using semantic search
    similar_title_enriched = None
    similar_title_index = None  # Store index of the input similar title to exclude it later
    if similar_title:
        similar_title_enriched = process_similar_title(similar_title, df_filtered, embeddings_filtered)
        
        # Find the exact similar title in the dataset to exclude it from results
        query_emb = embed_text([similar_title])
        title_similarities = cosine_similarity(query_emb, embeddings_filtered)[0]
        similar_title_index = np.argmax(title_similarities)  # Get the closest match
    
    # Calculate similar title similarity
    if similar_title_enriched:
        similar_emb = embed_text([similar_title_enriched])
        similar_sims = cosine_similarity(similar_emb, embeddings_filtered)[0]
    else:
        # If no similar title, use uniform similarity
        similar_sims = np.zeros(len(df_filtered))
    
    # Normalize genre weights to [0.67, 1.0] range
    # Original weights: [1.0, 1.5]
    genre_normalized_weights = np.ones(len(df_filtered))
    
    if use_weights:
        weights = calculate_likert_weights(
            action_intensity, narrative_complexity,
            darkness, realism
        )
        
        # Apply weights to genres (not category)
        # Each movie can have multiple genres separated by commas
        for i, (idx, row) in enumerate(df_filtered.iterrows()):
            genre_str = str(row.get('Genre', ''))
            genres_list = [g.strip() for g in genre_str.split(',')]
            
            # Find maximum weight among all genres of this movie
            max_weight = 1.0
            for genre in genres_list:
                if genre in weights:
                    max_weight = max(max_weight, weights[genre])
            
            # Normalize to 0-1 range (1.0-1.5 -> 0.67-1.0)
            genre_normalized_weights[i] = (max_weight - 0.5) / 1.0
    
    # NEW WEIGHTED FORMULA (all Hugging Face embeddings):
    # SCORE = (FREE_FORM × 0.25) + (SIMILAR_TITLE × 0.25) + (GENRE × 0.20) + (ENRICHED_QUERY × 0.30)
    final_scores = (
        (desc_sims * 0.25) +
        (similar_sims * 0.25) +
        (genre_normalized_weights * 0.20) +
        (full_sims * 0.30)
    )
    
    print(f"\n[Scoring] Formula: (DESC × 0.25) + (SIMILAR × 0.25) + (GENRE × 0.20) + (QUERY × 0.30)")
    print(f"[Scoring] All similarities computed using Hugging Face SentenceTransformer (no external APIs)")
    
    # Sort by scores (descending)
    sorted_indices = np.argsort(final_scores)[::-1]
    
    # Exclude the similar_title movie from recommendations if it was provided
    if similar_title_index is not None:
        sorted_indices = [idx for idx in sorted_indices if idx != similar_title_index]
    
    # Get top_k after exclusion
    top_indices = sorted_indices[:top_k]
    
    # Build results
    recommendations = []
    for idx in top_indices:
        row = df_filtered.iloc[idx]
        movie_title = str(row.get('Title', ''))
        movie_genre = str(row.get('Genre', ''))
        movie_desc = str(row.get('Description', ''))
        score = float(final_scores[idx])
        
        recommendations.append({
            "title": movie_title,
            "year": int(row.get('Year', 0)),
            "category": str(row.get('Category', '')),
            "genre": movie_genre,
            "blockid": str(row.get('BlockID', '')),
            "description": movie_desc,
            "score": score
        })
    
    # Generate GenAI-powered personalized justification
    genai_justification = None
    if enable_justification and len(recommendations) > 0:
        # Identify priority elements (lowest contributing factors)
        scores_breakdown = {
            "description_similarity": 0.25,
            "similar_title": 0.25,
            "genre_preference": 0.20,
            "enriched_query": 0.30
        }
        priority_elements = sorted(scores_breakdown.items(), key=lambda x: x[1])[:2]
        priority_elements = [elem[0] for elem in priority_elements]
        
        try:
            genai_justification = generate_recommendation_justification(
                user_preferences={
                    'description': description,
                    'similar_title': similar_title,
                    'action_intensity': action_intensity,
                    'narrative_complexity': narrative_complexity,
                    'darkness': darkness,
                    'realism': realism,
                    'period': period
                },
                recommendations=recommendations[:3],  # Top 3
                priority_elements=priority_elements
            )
        except Exception as e:
            print(f"GenAI justification error: {e}")
            genai_justification = None
    
    # Add genai_justification to each recommendation
    for rec in recommendations:
        rec['justification'] = genai_justification
    
    return {
        "genai_justification": genai_justification,
        "recommendations": recommendations
    }

if __name__ == "__main__":
    print("Building movie index...")
    df, emb = load_movie_index()
    print(f"Index built: {len(df)} movies/shows")
    
    # Test recommendation
    print("\n=== Recommendation Test ===")
    recs = recommend_movies(
        description="intense drama with suspense",
        action_intensity=4,
        narrative_complexity=4,
        darkness=4,
        realism=2,
        period="2010-2000",
        top_k=5
    )
    
    print("\nTop 5 recommendations:")
    for i, rec in enumerate(recs['recommendations'], 1):
        print(f"\n{i}. {rec['title']} ({rec['year']})")
        print(f"   Category: {rec['category']} | Genre: {rec['genre']}")
        print(f"   Score: {rec['score']:.4f}")
        print(f"   Description: {rec['description'][:100]}...")
        if rec.get('justification'):
            print(f"   Justification: {rec['justification']}")
