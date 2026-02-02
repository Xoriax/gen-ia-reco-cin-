"""
TMDB (The Movie Database) integration module for fetching movie posters and metadata.
Provides poster_path retrieval for movies to enhance UI display.
"""
import requests
from typing import Optional, Dict
from pathlib import Path
import json
import os
from datetime import datetime, timedelta

# TMDB Configuration
TMDB_BASE_URL = 'https://api.themoviedb.org/3'
TMDB_IMAGE_BASE = 'https://image.tmdb.org/t/p/w500'

# Cache file for TMDB results
TMDB_CACHE_FILE = Path(__file__).parent / '.tmdb_cache.json'
_CACHE = None
CACHE_EXPIRY_DAYS = 30

def _get_api_key() -> str:
    """
    Get TMDB API key from environment.
    Deferred loading to ensure .env is loaded by app.py first.
    """
    api_key = os.getenv('TMDB_API_KEY', '').strip()
    return api_key

def load_tmdb_cache() -> Dict:
    """Load TMDB cache from file."""
    global _CACHE
    if _CACHE is None:
        if TMDB_CACHE_FILE.exists():
            try:
                with open(TMDB_CACHE_FILE, 'r', encoding='utf-8') as f:
                    _CACHE = json.load(f)
            except:
                _CACHE = {}
        else:
            _CACHE = {}
    return _CACHE

def save_tmdb_cache():
    """Save TMDB cache to file."""
    global _CACHE
    if _CACHE is not None:
        try:
            with open(TMDB_CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(_CACHE, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[ERROR] Error saving TMDB cache: {e}")

def _is_cache_expired(timestamp: str) -> bool:
    """Check if cache entry is expired."""
    try:
        cache_time = datetime.fromisoformat(timestamp)
        expiry = cache_time + timedelta(days=CACHE_EXPIRY_DAYS)
        return datetime.now() > expiry
    except:
        return True

def fetch_movie_poster(
    movie_title: str,
    release_year: Optional[int] = None,
    use_cache: bool = True
) -> Optional[str]:
    """
    Fetch movie poster URL from TMDB API.
    
    Args:
        movie_title: Name of the movie
        release_year: Release year (optional, improves accuracy)
        use_cache: Use cached results when available
        
    Returns:
        Full poster image URL or None if not found
    """
    
    api_key = _get_api_key()
    if not api_key:
        print("[WARNING] TMDB_API_KEY not configured. Set TMDB_API_KEY environment variable.")
        return None
    
    # Build cache key
    cache_key = f"{movie_title}_{release_year}" if release_year else movie_title
    
    # Check cache
    if use_cache:
        cache = load_tmdb_cache()
        if cache_key in cache:
            cached_data = cache[cache_key]
            if not _is_cache_expired(cached_data.get('timestamp', '')):
                poster_path = cached_data.get('poster_path')
                if poster_path:
                    return f"{TMDB_IMAGE_BASE}{poster_path}"
                return None
    
    try:
        # Search strategy: try title alone first, then with year if needed
        search_query = movie_title
        search_params = {
            'api_key': api_key,
            'query': search_query,
            'language': 'en-US'
        }
        
        print(f"[TMDB] Searching for: {search_query}")
        search_response = requests.get(
            f"{TMDB_BASE_URL}/search/movie",
            params=search_params,
            timeout=5
        )
        
        if search_response.status_code != 200:
            print(f"[ERROR] TMDB search failed: {search_response.status_code}")
            return None
        
        search_data = search_response.json()
        results = search_data.get('results', [])
        
        # If no results found, try searching for partial title (useful for "Avatar: The Last Airbender" -> "The Last Airbender")
        if not results and ':' in movie_title:
            alt_title = movie_title.split(':')[1].strip()
            print(f"[TMDB] No results, trying alternate title: {alt_title}")
            search_params['query'] = alt_title
            search_response = requests.get(
                f"{TMDB_BASE_URL}/search/movie",
                params=search_params,
                timeout=5
            )
            search_data = search_response.json()
            results = search_data.get('results', [])
        
        if not results:
            print(f"[TMDB] [NO RESULTS] No results found for: {movie_title}")
            # Cache the negative result
            cache = load_tmdb_cache()
            cache[cache_key] = {
                'poster_path': None,
                'timestamp': datetime.now().isoformat()
            }
            save_tmdb_cache()
            return None
        
        # Get best match - prefer year match if available, then prefer results with posters
        best_match = results[0]
        
        # If year provided, try to match it more closely
        if release_year:
            year_matches = []
            for result in results:
                result_year = result.get('release_date', '')[:4]
                if result_year:
                    try:
                        if int(result_year) == release_year:
                            year_matches.append(result)
                    except ValueError:
                        pass
            
            # Use year match if found
            if year_matches:
                # Among year matches, prefer one with a poster
                matches_with_poster = [r for r in year_matches if r.get('poster_path')]
                if matches_with_poster:
                    best_match = matches_with_poster[0]
                    print(f"[TMDB] Matched by year ({release_year}) with poster: {best_match.get('title')}")
                else:
                    best_match = year_matches[0]
                    print(f"[TMDB] Matched by year ({release_year}): {best_match.get('title')}")
            else:
                # No year match found, prefer results with posters
                results_with_poster = [r for r in results if r.get('poster_path')]
                if results_with_poster:
                    best_match = results_with_poster[0]
                    print(f"[TMDB] No year match, using first result with poster: {best_match.get('title')}")
                else:
                    print(f"[TMDB] No year match and no posters available, using first result: {results[0].get('title')}")
        else:
            # No year provided, prefer results with posters
            results_with_poster = [r for r in results if r.get('poster_path')]
            if results_with_poster:
                best_match = results_with_poster[0]
                print(f"[TMDB] Using first result with poster: {best_match.get('title')}")
            else:
                print(f"[TMDB] No results with posters, using first result: {results[0].get('title')}")
        
        poster_path = best_match.get('poster_path')
        matched_title = best_match.get('title', movie_title)
        
        # Cache the result
        cache = load_tmdb_cache()
        cache[cache_key] = {
            'poster_path': poster_path,
            'tmdb_id': best_match.get('id'),
            'title': matched_title,
            'timestamp': datetime.now().isoformat()
        }
        save_tmdb_cache()
        
        if poster_path:
            poster_url = f"{TMDB_IMAGE_BASE}{poster_path}"
            print(f"[TMDB] [OK] Found poster for '{best_match.get('title')}'")
            return poster_url
        else:
            print(f"[TMDB] No poster available for: {movie_title}")
            return None
            
    except requests.exceptions.Timeout:
        print(f"[TMDB] TIMEOUT: Request timeout for: {movie_title}")
        return None
    except Exception as e:
        print(f"[TMDB] ERROR for '{movie_title}': {e}")
        return None

def fetch_multiple_posters(
    movies: list,
    use_cache: bool = True
) -> Dict[str, Optional[str]]:
    """
    Fetch posters for multiple movies.
    
    Args:
        movies: List of dicts with 'title' and optionally 'year'
        use_cache: Use cached results
        
    Returns:
        Dictionary mapping movie titles to poster URLs
    """
    results = {}
    for movie in movies:
        title = movie.get('title', '')
        year = movie.get('year')
        poster_url = fetch_movie_poster(title, year, use_cache)
        results[title] = poster_url
    return results

def get_tmdb_cache_stats() -> Dict:
    """Get statistics about cached TMDB data."""
    cache = load_tmdb_cache()
    return {
        'total_entries': len(cache),
        'file_path': str(TMDB_CACHE_FILE),
        'expires_in_days': CACHE_EXPIRY_DAYS
    }
