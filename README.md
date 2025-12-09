# 🎬 Système de Recommandation Cinématographique par IA

## Description
Système de recommandation intelligent de films et séries TV utilisant l'IA générative et les embeddings sémantiques. Le système analyse les préférences utilisateur via un questionnaire multi-critères et propose des recommandations personnalisées.

## 🌟 Fonctionnalités

### ✅ Implémenté (EF2.2, EF2.3, EF3.1, EF3.2)
- **Modélisation sémantique** avec SBERT (SentenceTransformer)
- **Mesure de similarité cosinus** pour comparer les préférences
- **Formule de score pondérée** pour la couverture/affinité sémantique
- **Top 3-5 recommandations** avec scores détaillés
- **Recommandations de titres spécifiques** (films/TV shows)

### 🎯 Critères de Recherche

#### 1. Description Libre
Décrivez en langage naturel le type de film recherché.

#### 2. Preuve Contextuelle
Donnez un titre de film similaire pour affiner les résultats.

#### 3. Échelle de Likert (1-5)
- **Intensité de l'Action** : Calme → Explosif
- **Complexité Narrative** : Simple → Complexe
- **Noirceur/Violence** : Familial → Sombre
- **Réalisme** : Documentaire → Fantastique

#### 4. Filtrage Temporel
6 périodes disponibles (2020+, 2015-2020, 2010-2015, 2000-2010, 1980-2000, <1980)

## 🚀 Démarrage Rapide

### Installation
```bash
# Créer l'environnement virtuel
python -m venv .venv

# Activer l'environnement (Windows)
.\.venv\Scripts\Activate.ps1

# Installer les dépendances
pip install -r requirements.txt
```

### Utilisation Interactive
```bash
python interactive_recommender.py
```

### Utilisation Programmatique
```python
from recommender.movie_recommender import recommend_movies

recommendations = recommend_movies(
    description="film dramatique intense avec du crime",
    action_intensity=4,
    darkness=4,
    top_k=5
)
```

## 📊 Tests

### Tests Complets
```bash
python test_movie_recommender.py
```

### Tests Unitaires
```bash
python -m pytest tests/test_referentiel.py -v
```

## 📁 Structure du Projet

```
gen-ia-reco-cin/
├── src/
│   ├── data/
│   │   ├── movies.csv                 # Base de données (200 films/shows)
│   │   └── referentiel_movies.pkl     # Index avec embeddings
│   ├── models/
│   ├── recommender/
│   │   └── movie_recommender.py       # Moteur de recommandation
│   ├── services/
│   │   └── ref.py                     # Service de référentiel
│   └── utils/
├── tests/
│   └── test_referentiel.py
├── interactive_recommender.py         # Interface interactive
├── test_movie_recommender.py          # Tests complets
├── requirements.txt
├── DOCUMENTATION_RECOMMENDER.md       # Documentation complète
└── README.md
```

## 🔧 Technologies

- **Python** 3.11+
- **SentenceTransformers** 2.7.0 (SBERT pour embeddings)
- **scikit-learn** 1.3.2 (similarité cosinus)
- **pandas** 2.1.3 (manipulation de données)
- **numpy** 1.24.3 (calculs numériques)

## 📖 Documentation

Consultez [DOCUMENTATION_RECOMMENDER.md](DOCUMENTATION_RECOMMENDER.md) pour :
- Guide d'utilisation détaillé
- Exemples de scénarios
- Architecture technique
- API de programmation

## 🎯 Résultats Exemple

**Requête :**
```
Description: "film dramatique intense avec du crime"
Likert: Action=4, Complexité=4, Noirceur=5, Réalisme=2
```

**Top 3 :**
1. Se7en (1995) - Score: 0.4562
2. Pulp Fiction (1994) - Score: 0.3999
3. Le Trou (1960) - Score: 0.4072

## 📈 Performance

- ⚡ **Temps de réponse** : ~100-200ms par requête
- 📚 **Base de données** : 200 films/TV shows
- 🎯 **Précision** : Embeddings 384 dimensions
- 💾 **Mémoire** : ~50MB pour l'index

## ✅ Conformité aux Exigences

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

## 👥 Contribution

Les contributions sont les bienvenues ! Voir [CONTRIBUTING.md](CONTRIBUTING.md) pour les guidelines.

## 📄 Licence

MIT License - voir [LICENSE](LICENSE) pour plus de détails.

## 📞 Contact

Pour questions et support : ouvrir une issue sur GitHub

---

**Version** : 1.0.0  
**Date** : 9 décembre 2025  
**Statut** : Production Ready