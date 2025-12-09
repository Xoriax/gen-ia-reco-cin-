from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pickle
import pandas as pd

MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
_MODEL: Optional[SentenceTransformer] = None
DEFAULT_CSV = Path(__file__).resolve().parents[1] / "data" / "movies.csv"
DEFAULT_PICKLE = Path(__file__).resolve().parents[1] / "data" / "referentiel_blockid.pkl"

def get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL

def load_blockids_from_csv(csv_path: str = None) -> List[str]:
    """
    Lit data/movies.csv et retourne la liste des BlockID uniques (ordre d'apparition).
    """
    p = Path(csv_path) if csv_path else DEFAULT_CSV
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    if "BlockID" not in df.columns:
        raise ValueError(f"'BlockID' column not found in {p}")
    # garder l'ordre d'apparition tout en dédupliquant
    seen = set()
    items = []
    for v in df["BlockID"].fillna("").astype(str).tolist():
        if v == "":
            continue
        if v not in seen:
            seen.add(v)
            items.append(v)
    return items

def build_embeddings(items: List[str]) -> np.ndarray:
    """Encode une liste de chaînes -> (n_items, dim)."""
    model = get_model()
    return model.encode(items, show_progress_bar=False, convert_to_numpy=True)

def build_and_save_from_csv(csv_path: str = None, out_path: str = None) -> Tuple[List[str], np.ndarray]:
    """
    Construit le référentiel (BlockID uniques + embeddings) depuis movies.csv
    et sauvegarde le résultat en pickle.
    """
    items = load_blockids_from_csv(csv_path)
    embeddings = build_embeddings(items)
    if out_path is None:
        out_path = str(DEFAULT_PICKLE)
    save_referentiel(out_path, items, embeddings)
    return items, embeddings

def save_referentiel(path: str, items: List[str], embeddings: np.ndarray) -> None:
    """Sauvegarde items + embeddings dans un fichier pickle."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump({"model": MODEL_NAME, "items": items, "embeddings": embeddings}, f)

def load_referentiel(path: str = None) -> Tuple[List[str], np.ndarray]:
    """Charge items et embeddings depuis un pickle. Si absent, construit depuis CSV."""
    p = Path(path) if path else DEFAULT_PICKLE
    if not p.exists():
        return build_and_save_from_csv()
    with p.open("rb") as f:
        data = pickle.load(f)
    return data["items"], data["embeddings"]

def embed_text(texts: List[str]) -> np.ndarray:
    """Encode une ou plusieurs chaînes."""
    model = get_model()
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

def recommend(query: str, items: List[str], embeddings: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Retourne top_k items (BlockID) les plus proches du query avec scores (cosine).
    Score dans [0,1] (1 = identique).
    """
    q_emb = embed_text([query])
    sims = cosine_similarity(q_emb, embeddings)[0]
    idx = np.argsort(sims)[::-1][:top_k]
    return [(items[i], float(sims[i])) for i in idx]

if __name__ == "__main__":
    # démonstration rapide (Windows: python .\src\services\referentiel.py)
    items, emb = load_referentiel()  # construira depuis src/data/movies.csv si nécessaire
    q = "film dramatique, intense, crime"
    recs = recommend(q, items, emb, top_k=3)
    print("Nombre de BlockID :", len(items))
    print("Query :", q)
    print("Top recommandations :", recs)