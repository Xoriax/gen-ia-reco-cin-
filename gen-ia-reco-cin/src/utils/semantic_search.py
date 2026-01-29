"""
Semantic search service using Hugging Face embeddings (SentenceTransformer).
Finds similar movies within our local dataset without external APIs.
"""
from typing import Optional, List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys
from pathlib import Path

# Add parent paths
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
_MODEL: Optional[SentenceTransformer] = None

def get_model() -> SentenceTransformer:
    """Get singleton SentenceTransformer instance."""
    global _MODEL
    if _MODEL is None:
        print("[HF] Loading SentenceTransformer model...")
        _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL

def embed_text(texts: List[str]) -> np.ndarray:
    """Encode text(s) to embeddings."""
    model = get_model()
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

def find_similar_movies_in_dataset(
    query_title: str,
    df,
    embeddings,
    top_k: int = 5,
    exclude_title: bool = True
) -> List[Dict]:
    """
    Find similar movies to a given title using semantic similarity.
    Works entirely locally - no API calls needed!
    
    Args:
        query_title: Movie title to find similar movies for
        df: Movies dataframe
        embeddings: Pre-computed embeddings for all movies
        top_k: Number of similar movies to return
        exclude_title: Exclude exact title matches
        
    Returns:
        List of similar movie data dictionaries
    """
    
    print(f"\n[SEMANTIC SEARCH] Finding movies similar to '{query_title}'")
    
    # Encode query title
    query_emb = embed_text([query_title])
    
    # Calculate similarity with all movies
    similarities = cosine_similarity(query_emb, embeddings)[0]
    
    # Get top indices
    top_indices = np.argsort(similarities)[::-1]
    
    similar_movies = []
    count = 0
    
    for idx in top_indices:
        if count >= top_k:
            break
        
        row = df.iloc[idx]
        movie_title = str(row.get('Film', ''))
        similarity = float(similarities[idx])
        
        # Optionally exclude exact matches or very similar titles
        if exclude_title and similarity > 0.95:
            continue
        
        similar_movies.append({
            "title": movie_title,
            "year": int(row.get('Année', 0)),
            "category": str(row.get('Catégorie', '')),
            "genre": str(row.get('Genre', '')),
            "description": str(row.get('Description narrative', '')),
            "similarity": similarity
        })
        count += 1
    
    print(f"   ✓ Found {len(similar_movies)} similar movies")
    for i, movie in enumerate(similar_movies[:3], 1):
        print(f"     {i}. {movie['title']} ({movie['year']}) - Similarity: {movie['similarity']:.1%}")
    
    return similar_movies

def aggregate_similar_movies_context(
    similar_movies: List[Dict],
    max_descriptions: int = 3
) -> str:
    """
    Aggregate descriptions from similar movies into enriched context.
    
    Args:
        similar_movies: List of similar movie data
        max_descriptions: Max descriptions to aggregate
        
    Returns:
        Aggregated text from similar movies
    """
    
    descriptions = [
        m.get('description', '') 
        for m in similar_movies[:max_descriptions]
        if m.get('description')
    ]
    
    genres = set()
    for m in similar_movies[:max_descriptions]:
        genre_str = m.get('genre', '')
        if genre_str:
            genres.update([g.strip() for g in genre_str.split(',')])
    
    # Combine into enriched context
    context_parts = descriptions.copy()
    if genres:
        context_parts.append(f"Common genres: {', '.join(list(genres)[:5])}")
    
    aggregated = " ".join(context_parts)
    return aggregated

def get_most_similar_movie(
    query_title: str,
    df,
    embeddings
) -> Optional[Dict]:
    """
    Get the single most similar movie to a query title.
    
    Args:
        query_title: Movie title to search for
        df: Movies dataframe
        embeddings: Pre-computed embeddings
        
    Returns:
        Single best match or None
    """
    similar = find_similar_movies_in_dataset(query_title, df, embeddings, top_k=1)
    return similar[0] if similar else None
