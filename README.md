# Système de Recommandation Cinématographique par IA

## Description

Système de recommandation intelligent de films et séries TV utilisant l'IA générative et les embeddings sémantiques. Le système analyse les préférences utilisateur via une interface web conversationnelle moderne et propose des recommandations personnalisées avec justifications contextuelles générées par IA.

### Points Forts

- **Recherche Sémantique Avancée** : Utilise SentenceTransformers pour comprendre le sens des requêtes en langage naturel
- **Enrichissement par IA Générative (Gemini)** : Les requêtes courtes sont enrichies par un vrai appel à Gemini (EF4.1), pas par un mapping statique
- **Interface Streamlit** : Questionnaire interactif + dashboard d'évaluation dans une seule application Python
- **Affiches TMDB** : Intégration automatique des posters de films via l'API TMDB
- **Justifications IA (Gemini)** : Explications personnalisées générées par Gemini pour chaque recommandation (EF4.3)
- **Évaluation des résultats générés** : Dashboard dédié (scores, avant/après, cache, latence, limites/risques)
- **Performance** : Cache local par usage (query, justification) et embeddings pré-calculés pour des réponses rapides

---

## Fonctionnalités Principales

### Critères de Recherche Multi-Dimensionnels

#### 1. Description Libre
Décrivez en langage naturel le type de film recherché. Le système comprend des descriptions comme :
- *"un film d'action intense avec des explosions"*
- *"comédie romantique légère et drôle"*
- *"thriller psychologique sombre et complexe"*

**Fonctionnalité EF4.1** : Les requêtes courtes (< 5 mots) sont automatiquement enrichies via un appel à l'API Gemini (avec cache et fallback local si la clé API est absente) pour améliorer la précision des résultats.

#### 2. Preuve Contextuelle (Titre Similaire)
Donnez un titre de film que vous avez aimé pour trouver des films similaires. Le système utilise la recherche sémantique pour identifier des œuvres comparables par leur thématique, ambiance et style narratif.

#### 3. Échelle de Likert (1-5)
Affinez vos préférences avec 4 dimensions cinématographiques :

- **Intensité de l'Action** : Calme (1) à Explosif (5)
- **Complexité Narrative** : Simple (1) à Labyrinthique (5)
- **Noirceur/Violence** : Familial (1) à Sombre (5)
- **Réalisme** : Documentaire (1) à Fantastique (5)

Ces critères sont convertis en pondérations thématiques qui influencent la sélection des films.

#### 4. Filtrage Temporel
Sélectionnez une période spécifique :
- **2020+** : Cinema contemporain
- **2015-2020** : Sorties récentes
- **2010-2015** : Classiques modernes
- **2000-2010** : Années 2000
- **1980-2000** : Classiques des années 80-90
- **< 1980** : Chefs-d'œuvre intemporels

### Système de Recommandation

- **Top 5 Recommandations** : Les meilleurs résultats classés par score de pertinence
- **Scores Détaillés** : Pourcentage d'affinité pour chaque recommandation
- **Métadonnées Complètes** : Titre, année, genre, catégorie, description
- **Affiches Visuelles** : Posters haute qualité via TMDB
- **Justification IA** : Explication personnalisée de pourquoi ces films correspondent à vos goûts

### Fonctionnalités Avancées (EF)

| Code | Fonctionnalité | Description |
|------|---------------|-------------|
| **EF2.2** | Embeddings Sémantiques | Modélisation avec SentenceTransformer (all-MiniLM-L12-v2) |
| **EF2.3** | Similarité Cosinus | Mesure mathématique de proximité sémantique |
| **EF3.1** | Score Pondéré | Formule combinant similarité sémantique et pondérations Likert |
| **EF3.2** | Top K Recommandations | Extraction des 5 meilleurs résultats |
| **EF4.1** | Enrichissement Requêtes | Expansion des requêtes courtes via l'API Gemini |
| **EF4.3** | Justifications GenAI | Synthèse narrative générée par Gemini, ancrée sur le top 3 SBERT |
| **C5.3** | Évaluation des résultats générés | Dashboard Streamlit : scores, avant/après, cache, latence, limites |

---

## Installation

### Prérequis

- **Python** : 3.11 ou supérieur
- **pip** : Gestionnaire de paquets Python
- **Clé API TMDB** : (Optionnelle) Pour afficher les posters de films

### Installation des Dépendances

```bash
# Cloner le repository
git clone <url-du-repo>
cd gen-ia-reco-cin

# Créer l'environnement virtuel
python -m venv .venv

# Activer l'environnement virtuel
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows CMD
.\.venv\Scripts\activate.bat

# Linux/Mac
source .venv/bin/activate

# Installer les dépendances
pip install -r gen-ia-reco-cin/requirements.txt
```

### Configuration des Clés API

Copiez `.env.example` vers `.env` à la racine du projet et complétez :

```env
TMDB_API_KEY=votre_cle_api_tmdb
GEMINI_API_KEY=votre_cle_api_gemini
```

- Clé TMDB (optionnelle, pour les posters) : [TMDB](https://www.themoviedb.org/settings/api)
- Clé Gemini (requise pour l'enrichissement EF4.1 et les justifications EF4.3) : [Google AI Studio](https://aistudio.google.com/)

Sans `GEMINI_API_KEY`, l'application fonctionne toujours (fallback local automatique) mais sans les appels GenAI réels. Voir le dashboard Évaluation pour vérifier si les appels partent bien vers l'API ou tombent en fallback.

---

## Utilisation

### Interface Web (Streamlit)

Lancez l'application Streamlit (questionnaire + dashboard d'évaluation) :

```bash
cd gen-ia-reco-cin
streamlit run app_streamlit.py
```

Ouvrez votre navigateur à l'adresse indiquée par Streamlit (par défaut **http://localhost:8501**).

**Onglet Recommandation :**
- Questionnaire hybride (texte libre, titre similaire, sliders Likert, période)
- Résultats avec posters, scores et justification Gemini

**Onglet Évaluation :**
- Distribution des scores, comparaison avant/après enrichissement Gemini
- Comparaison de réglages (température), cache hit rate, latence
- Grille d'évaluation manuelle des générations, limites & risques identifiés

### Utilisation Programmatique

```python
from src.recommender.movie_recommender import recommend_movies

# Recommandations basées sur une description
result = recommend_movies(
    description="film d'action intense avec des scènes explosives",
    action_intensity=5,
    narrative_complexity=3,
    darkness=4,
    realism=2,
    period="2020-2015",
    top_k=5,
    use_weights=True,
    enable_augmentation=True
)

# Afficher les résultats
for i, rec in enumerate(result['recommendations'], 1):
    print(f"{i}. {rec['title']} ({rec['year']}) - Score: {rec['score']:.1%}")

# Afficher la justification IA
print(f"\n{result['genai_justification']}")
```

### Recherche par Film Similaire

```python
from src.recommender.movie_recommender import recommend_movies

# Trouver des films similaires à The Dark Knight
recommendations = recommend_movies(
    similar_title="The Dark Knight",
    top_k=5
)

for rec in recommendations['recommendations']:
    print(f"• {rec['title']} ({rec['year']})")
```

---

## Tests

### Tests Complets

Exécutez le script de test interactif qui valide tous les scénarios :

```bash
cd gen-ia-reco-cin
python tests/test_movie_recommender.py
```

**Scénarios testés :**
- Description libre seule
- Titre similaire (preuve contextuelle)
- Échelle de Likert complète
- Filtrage par période
- Combinaison de tous les critères
- Enrichissement automatique (EF4.1)

### Tests Unitaires

```bash
# Tester le module de référentiel
python -m pytest gen-ia-reco-cin/tests/test_referentiel.py -v

# Tester l'augmentation de requêtes
python -m pytest gen-ia-reco-cin/tests/test_query_augmentation.py -v

# Tester les recommandations pondérées
python -m pytest gen-ia-reco-cin/tests/test_weighted_recommendation.py -v

# Tous les tests
python -m pytest gen-ia-reco-cin/tests/ -v
```

---

## Structure du Projet

```
gen-ia-reco-cin/
├── app_streamlit.py                    # Application Streamlit (front + dashboard)
├── requirements.txt                    # Dépendances Python
├── .streamlit/config.toml              # Thème Streamlit
│
├── src/                                # Code source
│   ├── data/
│   │   ├── movies.csv                  # Base de données (200+ films/séries)
│   │   ├── referentiel_movies.pkl      # Index avec embeddings pré-calculés
│   │   ├── referentiel_blockid.pkl     # Index des BlockID
│   │   ├── genai_calls_log.jsonl       # Log des appels Gemini (preuve pour évaluation)
│   │   └── genai_eval_grid.csv         # Grille d'évaluation manuelle (générée à l'usage)
│   │
│   ├── recommender/
│   │   └── movie_recommender.py        # Moteur de recommandation principal
│   │
│   ├── services/
│   │   ├── ref.py                      # Service de référentiel BlockID
│   │   ├── tmdb_service.py             # Intégration API TMDB (posters)
│   │   └── gemini_service.py           # Appel Gemini centralisé (cache, log, fallback)
│   │
│   ├── utils/
│   │   ├── query_augmentation.py       # Enrichissement via Gemini (EF4.1)
│   │   ├── semantic_search.py          # Recherche sémantique locale
│   │   └── genai_justification.py      # Justifications via Gemini (EF4.3)
│   │
│   └── evaluation/
│       └── metrics.py                  # Métriques et comparaisons pour le dashboard (C5.3)
│
└── tests/                              # Tests
    ├── test_movie_recommender.py       # Tests complets du système
    ├── test_referentiel.py             # Tests référentiel
    ├── test_query_augmentation.py      # Tests enrichissement
    ├── test_weighted_recommendation.py # Tests pondérations
    └── test_ef42.py                    # Tests fonctionnalité EF4.2
```

---

## Technologies Utilisées

### Backend & IA

| Technologie | Version | Usage |
|-------------|---------|-------|
| **Python** | 3.11+ | Langage principal |
| **SentenceTransformers** | 2.7.0 | Embeddings sémantiques (all-MiniLM-L12-v2) |
| **scikit-learn** | 1.3.2 | Calcul de similarité cosinus |
| **pandas** | 2.1.3 | Manipulation de données |
| **numpy** | 1.24.3 | Calculs numériques |
| **PyTorch** | 2.1.1 | Backend pour transformers |
| **google-genai** | 0.7.0 | SDK officiel Gemini (EF4.1, EF4.3) |

### Frontend & Évaluation

| Technologie | Usage |
|-------------|-------|
| **Streamlit** | Interface web (questionnaire + dashboard) |
| **Plotly** | Graphiques du dashboard d'évaluation |
| **TMDB API** | Récupération des posters |

### Tests & Développement

| Technologie | Usage |
|-------------|-------|
| **pytest** | Framework de tests |
| **pytest-cov** | Couverture de code |
| **python-dotenv** | Gestion variables d'environnement |
| **requests** | Requêtes HTTP (TMDB) |

---

## Base de Données

### Contenu

- **200+ films et séries TV** référencés
- **Métadonnées complètes** : Titre, année, genre, catégorie, description narrative
- **Classifications thématiques** : BlockID pour catégorisation fine
- **Embeddings pré-calculés** : Performance optimale (pas de calcul à la volée)

### Champs Disponibles

| Champ | Description |
|-------|-------------|
| `Film` | Titre du film/série |
| `Année` | Année de sortie |
| `Genre` | Genre principal |
| `Catégorie` | Film ou TV Show |
| `Description narrative` | Synopsis détaillé |
| `BlockID` | Identifiant thématique |

---

## Algorithme de Recommandation

### Formule de Score

Le score final combine plusieurs composantes :

```
Score Final = (α × Similarité_Sémantique) + (β × Score_Likert)

où :
- Similarité_Sémantique : Cosinus entre embedding requête et embedding film
- Score_Likert : Pondération basée sur les critères Likert
- α, β : Coefficients de balance (configurables)
```

### Pipeline de Recommandation

```
1. Entrée Utilisateur
2. Enrichissement Requête (EF4.1) si nécessaire
3. Génération Embedding Requête
4. Calcul Similarité avec Base de Données
5. Application Filtres (période, pondérations)
6. Tri et Sélection Top K
7. Génération Justification IA (EF4.3)
8. Récupération Posters TMDB
9. Retour Résultats
```

---

## Configuration Avancée

### Paramètres du Modèle

Modifiez dans [movie_recommender.py](gen-ia-reco-cin/src/recommender/movie_recommender.py) :

```python
MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"  # Modèle d'embeddings
DEFAULT_TOP_K = 5                                        # Nombre de recommandations
AUGMENTATION_WORD_THRESHOLD = 5                          # Seuil enrichissement
```

### Coefficients de Pondération

Ajustez les poids des critères Likert dans `calculate_likert_weights()` :

```python
weights = {
    'action': action_intensity * 0.3,
    'complexity': narrative_complexity * 0.25,
    'darkness': darkness * 0.25,
    'realism': realism * 0.2
}
```

---

## Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. **Fork** le projet
2. Créez une **branche feature** (`git checkout -b feature/AmazingFeature`)
3. **Committez** vos changements (`git commit -m 'Add AmazingFeature'`)
4. **Push** sur la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une **Pull Request**

---

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---

## Auteurs

Développé pour explorer les capacités de l'IA dans la recommandation cinématographique.

---

## Support

Pour toute question ou problème :
- Ouvrez une **Issue** sur GitHub
- Contactez l'équipe de développement

---

## Roadmap

### Fonctionnalités Futures

- [ ] **Multi-utilisateurs** : Profils et historique personnalisés
- [ ] **Système de notation** : Permettre aux utilisateurs de noter les recommandations
- [ ] **Apprentissage continu** : Amélioration des recommandations basée sur le feedback
- [ ] **Export de listes** : Exporter les recommandations en PDF/CSV
- [ ] **Recommandations de groupe** : Trouver des films adaptés à plusieurs personnes
- [ ] **Intégration streaming** : Liens directs vers Netflix, Prime, etc.
- [ ] **Mode sombre** : Interface avec thème sombre
- [ ] **Multilingue** : Support de plusieurs langues

---

## Ressources & Références

- [SentenceTransformers Documentation](https://www.sbert.net/)
- [TMDB API Documentation](https://developers.themoviedb.org/3)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Gemini API Documentation](https://ai.google.dev/gemini-api/docs)
- [scikit-learn Documentation](https://scikit-learn.org/)

---

<div align="center">

**Si vous aimez ce projet, n'hésitez pas à lui donner une étoile !**

Made by AI Enthusiasts

</div>

---

## Résultats Exemple

**Requête :**
```
Description: "film dramatique intense avec du crime"
Likert: Action=4, Complexité=4, Noirceur=5, Réalisme=2
```

**Top 3 :**
1. Se7en (1995) - Score: 0.4562
2. Pulp Fiction (1994) - Score: 0.3999
3. Le Trou (1960) - Score: 0.4072

## Performance

- Base de données : 200 films/TV shows
- Précision : Embeddings 384 dimensions
- Mémoire : ~50MB pour l'index
- Latence Gemini (EF4.1/EF4.3) : voir dashboard Évaluation (mesurée en direct par appel)

---

## IA Générative - Cas d'usage & Évaluation

Cette section reprend la structure conseillée par la grille de notation "Projet 3 : IA Générative" (C5.1-C5.3), pour servir de base au support de présentation.

### Cas d'usage
1. **Enrichissement de requête (EF4.1)** : quand l'utilisateur saisit une description trop courte (< 5 mots, ex. "action"), le système manque de signal sémantique pour un bon matching SBERT. Gemini reformule la requête en une phrase riche en mots-clés cinématographiques (genre, ambiance, ton), utilisée ensuite pour l'embedding.
2. **Justification de recommandation (EF4.3)** : une fois le top 3 sélectionné par le scoring SBERT, Gemini rédige une synthèse narrative qui explique pourquoi ces films correspondent au profil Likert/texte libre de l'utilisateur, apportant une valeur pédagogique et UX sans jamais influencer le choix des films eux-mêmes.

### Choix du modèle et de l'approche
- **Modèle** : `gemini-2.0-flash` (rapide, faible coût, adapté à un usage limité à 1 appel par sortie).
- **Grounding** : le prompt de justification liste explicitement les films déjà sélectionnés par SBERT. Gemini commente, il ne recommande pas, ce qui réduit le risque d'hallucination.
- **Sobriété** : cache local par usage (`.query_cache.json`, `.justification_cache.json`) + fallback local automatique si la clé API est absente ou l'appel échoue, pour respecter la contrainte "une seule requête API par type de sortie" et ne jamais bloquer la démo.

### Solution développée
- `src/services/gemini_service.py` centralise tous les appels (cache, mesure de latence, log JSONL, fallback).
- `src/utils/query_augmentation.py` (EF4.1) et `src/utils/genai_justification.py` (EF4.3) construisent les prompts et consomment ce service.
- Chaque appel (prompt, réponse, latence, cache hit, fallback) est journalisé dans `src/data/genai_calls_log.jsonl`, source de données du dashboard d'évaluation.

### Évaluation des résultats
Dashboard Streamlit (onglet "Évaluation") :
- distribution des scores de la recommandation courante ;
- comparaison **avant/après** activation de l'enrichissement Gemini (delta de score sur le top résultat) ;
- comparaison de **réglages** (température 0.2 vs 0.9) sur un même prompt, avec longueur/latence des deux réponses ;
- taux de cache hit vs appels API réels vs fallback, et statistiques de latence ;
- grille d'évaluation manuelle (prompt/réponse/note 1-5) sur les dernières générations réelles ;
- section **Limites & risques** : hallucination (mitigée par le grounding), dépendance/coût API, latence, biais du modèle, absence de garantie factuelle sur le texte généré.

### Valeur métier
L'enrichissement et la justification GenAI transforment un moteur de similarité "boîte noire" en système explicable : l'utilisateur comprend *pourquoi* un film lui est proposé, et une requête vague ("un truc sombre") produit quand même un résultat pertinent grâce à la reformulation Gemini. Ce sont deux leviers directs d'adoption et de confiance pour un produit de recommandation grand public.

---

## Conformité aux Exigences

| Exigence | Statut | Description |
|----------|--------|-------------|
| **EF2.2** | OK | Modélisation sémantique (SBERT) |
| **EF2.3** | OK | Mesure de similarité cosinus |
| **EF3.1** | OK | Formule de score pondérée |
| **EF3.2** | OK | Top 3-5 recommandations |
| **EF4.1** | OK | Enrichissement de requête via un vrai appel API Gemini (plus de mapping statique) |
| **EF4.3** | OK | Justification générée par Gemini, ancrée sur le top 3 SBERT |
| **C5.3** | OK | Dashboard d'évaluation des résultats générés (scores, avant/après, cache, latence, limites) |
| **Titres spécifiques** | OK | Recommandations de films/shows réels |
| **Description libre** | OK | Analyse de texte naturel |
| **Titre similaire** | OK | Preuve contextuelle |
| **Échelle Likert** | OK | 4 dimensions (1-5) |
| **Filtrage temporel** | OK | 6 périodes disponibles |

---