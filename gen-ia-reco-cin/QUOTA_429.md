# EF4.1 : Gestion de l'erreur 429 (Quota API dépassé)

## Problème
L'erreur 429 signifie que vous avez dépassé le quota gratuit de l'API Gemini :
- **15 requêtes par minute**
- **1 500 requêtes par jour**

## Solutions implémentées

### 1. **Cache automatique** (Recommandé)
Le système enregistre automatiquement les requêtes enrichies dans un cache local.

**Comment ça marche :**
```python
from query_augmentation import augment_query_with_gemini

# Première fois : appel API
enriched = augment_query_with_gemini("action film")  # ✨ Appel Gemini

# Fois suivantes : utilisation du cache
enriched = augment_query_with_gemini("action film")  # 📦 Cache utilisé
```

**Cache pré-rempli :**
Le fichier `.query_cache.json` contient déjà 7 requêtes courantes :
- action film
- comédie
- thriller sombre
- animation japonaise
- film romantique
- horreur
- science-fiction

### 2. **Désactiver l'enrichissement temporairement**
```python
from movie_recommender import recommend_movies

# Désactiver EF4.1
recommendations = recommend_movies(
    description="action film",
    enable_augmentation=False  # Pas d'appel API
)
```

### 3. **Attendre avant de réessayer**
Si vous voyez l'erreur 429 :
- **Attendez 1 minute** pour le quota par minute
- **Attendez le lendemain** si quota journalier atteint

## Vérifier votre quota

1. Allez sur https://aistudio.google.com/app/apikey
2. Cliquez sur votre clé API
3. Consultez "Usage"

## Augmenter le quota

Pour plus de requêtes, passez à un plan payant :
- https://ai.google.dev/pricing
- Tarification : ~$0.001 par requête

## Fichiers modifiés

- `src/utils/query_augmentation.py` : Ajout du cache
- `src/utils/.query_cache.json` : Cache des requêtes enrichies
- `.gitignore` : Exclusion du cache du versionning
