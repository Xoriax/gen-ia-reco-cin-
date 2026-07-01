"""
Rebuilds gen-ia-reco-cin/src/data/movies.csv (+ a Parquet twin) from TMDB,
pulling up to MOVIE_TARGET + TV_TARGET titles (top_rated endpoints, same
approach as the original script, just paginated further and hardened).

Requires TMDB_BEARER_TOKEN in .env (TMDB v4 auth, distinct from the
TMDB_API_KEY used elsewhere in the project for the v3 query-param auth).
"""
import csv
import os
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

TOKEN = os.getenv("TMDB_BEARER_TOKEN", "").strip()
if not TOKEN:
    sys.exit("TMDB_BEARER_TOKEN is missing. Add it to .env (see .env.example).")

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "accept": "application/json",
}

BASE_URL = "https://api.themoviedb.org/3"
TMDB_MAX_PAGE = 500  # hard limit enforced by TMDB's API

MOVIE_TARGET = 10000
TV_TARGET = 10000

OUTPUT_CSV = ROOT / "gen-ia-reco-cin" / "src" / "data" / "movies.csv"
OUTPUT_PARQUET = ROOT / "gen-ia-reco-cin" / "src" / "data" / "movies.parquet"
STALE_INDEX_FILES = ["referentiel_movies.pkl", "referentiel_blockid.pkl"]

FIELDNAMES = ["FilmID", "BlockID", "Catégorie", "Genre", "genre_ids", "Année", "Film", "Description narrative"]


_PUNCT_REPLACEMENTS = {
    "–": "-", "—": "-", "−": "-",  # en dash, em dash, minus sign
    "‘": "'", "’": "'",                  # curly single quotes
    "“": '"', "”": '"',                  # curly double quotes
    "…": "...",                               # ellipsis
}


def remove_accents(text):
    """
    Strips accents (café -> cafe) and normalizes typographic punctuation to
    ASCII equivalents. Replacing dashes/quotes with spaced ASCII characters
    (instead of just deleting them) avoids fusing words together, e.g.
    "threat--one that" instead of "threatone that".
    """
    if not isinstance(text, str):
        return text
    for char, replacement in _PUNCT_REPLACEMENTS.items():
        text = text.replace(char, replacement)
    nfkd_form = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd_form if not unicodedata.combining(c) and ord(c) < 128)


def api_get(url, params=None, max_retries=5):
    """GET with retry/backoff on rate limiting (429) and transient errors."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
        except requests.exceptions.RequestException as e:
            wait = 2 ** attempt
            print(f"  network error ({e}), retrying in {wait}s...")
            time.sleep(wait)
            continue

        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", 1))
            time.sleep(retry_after + 0.5)
            continue

        if resp.status_code >= 500:
            wait = 2 ** attempt
            print(f"  server error {resp.status_code}, retrying in {wait}s...")
            time.sleep(wait)
            continue

        resp.raise_for_status()
        return resp.json()

    raise RuntimeError(f"Too many failed retries for {url}")


def get_genres_map(media_type):
    data = api_get(f"{BASE_URL}/genre/{media_type}/list", params={"language": "en-US"})
    return {g["id"]: g["name"] for g in data.get("genres", [])}


def get_top_items(endpoint, category, genres_map, target_count):
    """
    Paginates a TMDB top_rated endpoint until target_count usable items are
    collected (or TMDB's page limit is reached). Items without a text
    description are skipped since they are unusable for SBERT embeddings.
    """
    items = []
    seen_ids = set()
    page = 1

    while len(items) < target_count and page <= TMDB_MAX_PAGE:
        data = api_get(f"{BASE_URL}/{endpoint}", params={"language": "en-US", "page": page})
        results = data.get("results", [])
        if not results:
            break

        for result in results:
            item_id = result.get("id")
            if item_id in seen_ids:
                continue

            overview = (result.get("overview") or "").strip()
            if not overview:
                continue  # no usable text for semantic embeddings

            seen_ids.add(item_id)
            genre_ids = result.get("genre_ids", [])
            genre_names = [genres_map.get(gid, "") for gid in genre_ids if gid in genres_map]
            all_genres = ", ".join(genre_names)
            block_id = genre_names[0] if genre_names else ""
            date_field = result.get("release_date") or result.get("first_air_date")
            year = date_field[:4] if date_field and len(date_field) >= 4 else ""

            items.append({
                "FilmID": item_id,
                "BlockID": remove_accents(block_id),
                "Catégorie": remove_accents(category),
                "Genre": remove_accents(all_genres),
                "genre_ids": ",".join(str(gid) for gid in genre_ids),
                "Année": year,
                "Film": remove_accents(result.get("title") or result.get("name") or ""),
                "Description narrative": remove_accents(overview),
            })

            if len(items) >= target_count:
                break

        if page % 25 == 0 or len(items) >= target_count:
            print(f"  {category}: page {page}/{min(data.get('total_pages', TMDB_MAX_PAGE), TMDB_MAX_PAGE)} "
                  f"-> {len(items)}/{target_count} collected")

        page += 1
        time.sleep(0.05)  # polite pacing, well under TMDB's rate limits

    return items


def main():
    print("Fetching genre maps...")
    movie_genres = get_genres_map("movie")
    tv_genres = get_genres_map("tv")

    print(f"Fetching up to {MOVIE_TARGET} movies (movie/top_rated)...")
    movies = get_top_items("movie/top_rated", "Film", movie_genres, MOVIE_TARGET)
    print(f"  -> {len(movies)} movies collected")

    print(f"Fetching up to {TV_TARGET} TV shows (tv/top_rated)...")
    tvshows = get_top_items("tv/top_rated", "TVShow", tv_genres, TV_TARGET)
    print(f"  -> {len(tvshows)} TV shows collected")

    all_items = movies + tvshows
    df = pd.DataFrame(all_items, columns=FIELDNAMES)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    df.to_parquet(OUTPUT_PARQUET, index=False)

    csv_size_mb = OUTPUT_CSV.stat().st_size / (1024 * 1024)
    parquet_size_mb = OUTPUT_PARQUET.stat().st_size / (1024 * 1024)

    print(f"\nDone: {len(df)} items written")
    print(f"  CSV     : {OUTPUT_CSV} ({csv_size_mb:.1f} MB)")
    print(f"  Parquet : {OUTPUT_PARQUET} ({parquet_size_mb:.1f} MB)")

    for stale_name in STALE_INDEX_FILES:
        stale_path = OUTPUT_CSV.parent / stale_name
        if stale_path.exists():
            stale_path.unlink()
            print(f"  Removed stale index {stale_name} (will rebuild automatically on next app start)")


if __name__ == "__main__":
    main()
