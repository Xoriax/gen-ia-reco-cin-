# Système de Recommandation Cinématographique par IA

## Description

Système de recommandation intelligent de films et séries TV utilisant l'IA générative et les embeddings sémantiques. Le système analyse les préférences utilisateur via une interface web conversationnelle moderne et propose des recommandations personnalisées avec justifications contextuelles générées par IA.

### Points Forts

- **Recherche Sémantique Avancée** : Utilise SentenceTransformers pour comprendre le sens des requêtes en langage naturel
- **Enrichissement Automatique** : Augmentation intelligente des requêtes courtes pour améliorer la précision
- **Interface Conversationnelle** : Questionnaire interactif étape par étape avec retour en arrière
- **Affiches TMDB** : Intégration automatique des posters de films via l'API TMDB
- **Justifications IA** : Explications personnalisées générées par IA pour chaque recommandation
- **Performance** : Cache intelligent et embeddings pré-calculés pour des réponses rapides

---

## Fonctionnalités Principales

### Critères de Recherche Multi-Dimensionnels

#### 1. Description Libre
Décrivez en langage naturel le type de film recherché. Le système comprend des descriptions comme :
- *"un film d'action intense avec des explosions"*
- *"comédie romantique légère et drôle"*
- *"thriller psychologique sombre et complexe"*

**Fonctionnalité EF4.1** : Les requêtes courtes (< 5 mots) sont automatiquement enrichies avec un contexte sémantique pour améliorer la précision des résultats.

#### 2. Preuve Contextuelle (Titre Similaire)
Donnez un titre de film que vous avez aimé pour trouver des films similaires. Le système utilise la recherche sémantique pour identifier des œuvres comparables par leur thématique, ambiance et style narratif.

#### 3. Échelle de Likert (1-5)
Affinez vos préférences avec 4 dimensions cinématographiques :

- **Intensité de l'Action** : Calme (1) → Explosif (5)
- **Complexité Narrative** : Simple (1) → Labyrinthique (5)
- **Noirceur/Violence** : Familial (1) → Sombre (5)
- **Réalisme** : Documentaire (1) → Fantastique (5)

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
| **EF4.1** | Enrichissement Requêtes | Expansion automatique des requêtes courtes |
| **EF4.3** | Justifications GenAI | Génération de textes explicatifs personnalisés |

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

### Configuration TMDB (Optionnelle)

Pour afficher les posters de films, créez un fichier `.env` dans le dossier `gen-ia-reco-cin/` :

```env
TMDB_API_KEY=votre_cle_api_tmdb
```

Obtenez une clé API gratuite sur [TMDB](https://www.themoviedb.org/settings/api)

---

## Utilisation

### Interface Web (Recommandé)

Lancez l'application Flask avec l'interface conversationnelle interactive :

```bash
cd gen-ia-reco-cin
python app.py
```

Ouvrez votre navigateur à l'adresse : **http://localhost:5000**

**Interface Conversationnelle :**
- Questions posées une par une
- Possibilité de revenir en arrière
- Visualisation de l'historique des réponses
- Résultats avec posters et justifications IA

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
├── app.py                              # Application Flask principale
├── requirements.txt                    # Dépendances Python
│
├── src/                                # Code source
│   ├── data/
│   │   ├── movies.csv                  # Base de données (200+ films/séries)
│   │   ├── referentiel_movies.pkl      # Index avec embeddings pré-calculés
│   │   └── referentiel_blockid.pkl     # Index des BlockID
│   │
│   ├── recommender/
│   │   └── movie_recommender.py        # Moteur de recommandation principal
│   │
│   ├── services/
│   │   ├── ref.py                      # Service de référentiel BlockID
│   │   └── tmdb_service.py             # Intégration API TMDB (posters)
│   │
│   └── utils/
│       ├── query_augmentation.py       # Enrichissement automatique (EF4.1)
│       ├── semantic_search.py          # Recherche sémantique locale
│       └── genai_justification.py      # Génération justifications (EF4.3)
│
├── templates/                          # Templates HTML
│   ├── index.html                      # Page d'accueil
│   └── interactive.html                # Interface conversationnelle
│
├── static/                             # Assets statiques
│   ├── style.css                       # Styles globaux
│   ├── conversation.css                # Styles conversationnels
│   ├── script.js                       # Scripts frontend
│   └── conversation.js                 # Logique conversationnelle
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
| **Flask** | 3.0.0 | Framework web |
| **SentenceTransformers** | 2.7.0 | Embeddings sémantiques (all-MiniLM-L12-v2) |
| **scikit-learn** | 1.3.2 | Calcul de similarité cosinus |
| **pandas** | 2.1.3 | Manipulation de données |
| **numpy** | 1.24.3 | Calculs numériques |
| **PyTorch** | 2.1.1 | Backend pour transformers |

### Frontend

| Technologie | Usage |
|-------------|-------|
| **HTML5** | Structure |
| **CSS3** | Styles personnalisés |
| **JavaScript (Vanilla)** | Interactivité |
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

```mermaid
1. Entrée Utilisateur
   ↓
2. Enrichissement Requête (EF4.1) si nécessaire
   ↓
3. Génération Embedding Requête
   ↓
4. Calcul Similarité avec Base de Données
   ↓
5. Application Filtres (période, pondérations)
   ↓
6. Tri et Sélection Top K
   ↓
7. Génération Justification IA (EF4.3)
   ↓
8. Récupération Posters TMDB
   ↓
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
- [Flask Documentation](https://flask.palletsprojects.com/)
- [scikit-learn Documentation](https://scikit-learn.org/)

---

<div align="center">

**Si vous aimez ce projet, n'hésitez pas à lui donner une étoile !**

Made by AI Enthusiasts

</div>
- SentenceTransformers 2.7.0 (SBERT pour embeddings)
- scikit-learn 1.3.2 (similarité cosinus)
- pandas 2.1.3 (manipulation de données)
- numpy 1.24.3 (calculs numériques)

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

- Temps de réponse : ~100-200ms par requête
- Base de données : 200 films/TV shows
- Précision : Embeddings 384 dimensions
- Mémoire : ~50MB pour l'index

## Conformité aux Exigences

| Exigence | Statut | Description |
|----------|--------|-------------|
| **EF2.2** | ✅ | Modélisation sémantique (SBERT) |
| **EF2.3** | ✅ | Mesure de similarité cosinus |
| **EF3.1** | ✅ | Formule de score pondérée |
| **EF3.2** | ✅ | Top 3-5 recommandations |
| **Titres spécifiques** | ✅ | Recommandations de films/shows réels |
| **Description libre** | ✅ | Analyse de texte naturel |
| **Titre similaire** | ✅ | Preuve contextuelle |
| **Échelle Likert** | ✅ | 4 dimensions (1-5) |
| **Filtrage temporel** | ✅ | 6 périodes disponibles |
---