import importlib.util
from pathlib import Path
import pytest
import numpy as np

# charge dynamiquement referentiel.py ou ref.py (selon ce qui existe)
SERVICES_DIR = Path(__file__).resolve().parents[1] / "src" / "services"
MODULE_PATH = None

for fname in ("referentiel.py", "ref.py"):
    p = SERVICES_DIR / fname
    if p.exists():
        MODULE_PATH = p
        break

if MODULE_PATH is None:
    pytest.skip("Aucun module referentiel.py ou ref.py trouvé dans src/services", allow_module_level=True)

spec = importlib.util.spec_from_file_location("service_mod", MODULE_PATH)
service = importlib.util.module_from_spec(spec)
spec.loader.exec_module(service)

def test_load_blockids_from_csv_returns_list():
    """Teste que load_blockids_from_csv retourne une liste non vide."""
    items = service.load_blockids_from_csv()
    assert isinstance(items, list), "load_blockids_from_csv doit retourner une liste"
    assert len(items) > 0, "La liste de BlockID ne doit pas être vide"
    assert all(isinstance(x, str) for x in items), "Tous les BlockID doivent être des chaînes"
    assert "Drama" in items, "Drama doit être dans les BlockID"

def test_build_embeddings_shape():
    """Teste que build_embeddings retourne un array de la bonne forme."""
    test_items = ["Drama", "Animation", "Comedy"]
    embeddings = service.build_embeddings(test_items)
    assert isinstance(embeddings, np.ndarray), "embeddings doit être un numpy array"
    assert embeddings.shape[0] == len(test_items), "Nombre de lignes == nombre d'items"
    assert embeddings.shape[1] > 0, "Nombre de dimensions > 0"
    assert np.isfinite(embeddings).all(), "Tous les embeddings doivent être finis (pas NaN/inf)"

def test_build_and_save_and_load_tmp(tmp_path):
    """Teste build_and_save_from_csv, save et load_referentiel."""
    out = tmp_path / "referentiel_test.pkl"
    items, embeddings = service.build_and_save_from_csv(out_path=str(out))
    assert isinstance(items, list) and len(items) > 0, "items doit être une liste non vide"
    assert hasattr(embeddings, "shape"), "embeddings doit avoir un attribut shape"
    assert embeddings.shape[0] == len(items), "Nombre de lignes == nombre d'items"
    loaded_items, loaded_embeddings = service.load_referentiel(path=str(out))
    assert loaded_items == items, "Les items chargés doivent correspondre aux originaux"
    assert loaded_embeddings.shape == embeddings.shape, "La forme des embeddings doit être identique"
    assert np.isfinite(loaded_embeddings).all(), "Tous les embeddings chargés doivent être finis"

def test_embed_text_encoding():
    """Teste que embed_text encode correctement le texte."""
    texts = ["drama film", "animation movie"]
    encoded = service.embed_text(texts)
    assert isinstance(encoded, np.ndarray), "embed_text doit retourner un numpy array"
    assert encoded.shape[0] == len(texts), "Nombre de lignes == nombre de textes"
    assert encoded.shape[1] > 0, "Nombre de dimensions > 0"

def test_recommend_returns_top_k_with_scores(tmp_path):
    """Teste que recommend retourne top_k résultats avec scores valides."""
    out = tmp_path / "referentiel_test.pkl"
    items, embeddings = service.build_and_save_from_csv(out_path=str(out))
    recs = service.recommend("drama", items, embeddings, top_k=3)
    assert isinstance(recs, list), "recommend doit retourner une liste"
    assert len(recs) <= 3, "Nombre de résultats <= top_k"
    for label, score in recs:
        assert isinstance(label, str), "Label doit être une chaîne"
        assert isinstance(score, float), "Score doit être un float"
        assert -1.0 <= score <= 1.0, f"Score doit être entre -1 et 1, reçu {score}"
    assert any(r[0] == "Drama" for r in recs), f"'Drama' non trouvé dans {recs}"

def test_recommend_respects_top_k():
    """Teste que recommend respecte le paramètre top_k."""
    items, embeddings = service.load_referentiel()
    for top_k in [1, 3, 5]:
        recs = service.recommend("film", items, embeddings, top_k=top_k)
        assert len(recs) <= top_k, f"Nombre de résultats doit être <= {top_k}"