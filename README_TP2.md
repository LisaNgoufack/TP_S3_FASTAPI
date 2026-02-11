# TP2 - API EDA : Statistiques Descriptives & Graphiques

API REST FastAPI pour l'analyse exploratoire de données (EDA) avec génération de statistiques et visualisations Plotly.

## Objectifs

Produire des résumés statistiques et des graphiques exploitables sans notebook, pour analyser la qualité et les caractéristiques des données.

## Fonctionnalités

- Statistiques descriptives : count, mean, std, quantiles, missing_rate
- Agrégations par groupe : mean, median, sum, count
- Matrice de corrélation : Pearson + top paires corrélées
- Visualisations Plotly : histogramme, boxplot, barplot (format JSON)

## Installation

### Prérequis
- Python 3.12+
- pip

### Installation des dépendances
bash
pip install fastapi uvicorn pandas numpy pydantic plotly


##  Démarrage
bash
uvicorn app.main:app --reload


L'API sera accessible sur : http://127.0.0.1:8000

Documentation interactive : http://127.0.0.1:8000/docs

## Structure du projet

TP2_API_EDA/
├── app/
│   ├── main.py                 # Point d'entrée FastAPI
│   ├── routers/
│   │   └── eda.py              # Endpoints EDA
│   └── services/
│       └── eda_service.py      # Logique métier EDA
├── README_TP2.md
└── TP2_RAPPORT.adoc
```

## Endpoints

### 1. POST /eda/dataset-generate

Génère un dataset pour analyse exploratoire.

Request:
json
{
  "phase": "eda",
  "seed": 42,
  "n": 100
}


Variables générées :
- Numériques : age, income, spend, visits
- Catégorielles : segment (A/B/C), channel (web/store/app), churn (0/1)
- Défauts injectés : NA (5-10%), outliers (1-2% sur income)

### 2. POST `/eda/summary`

Statistiques descriptives par variable.

Request:
json
{
  "dataset_id": "uuid"
}


Response (extrait):
json
{
  "result": {
    "statistics": {
      "age": {
        "count": 91,
        "mean": 44.8,
        "std": 14.2,
        "min": 20,
        "25%": 34,
        "50%": 44,
        "75%": 57,
        "max": 68,
        "missing_rate": 9
      },
      "segment": {
        "unique": 3,
        "top": "A",
        "freq": 36,
        "distribution": {"A": 36, "B": 36, "C": 28}
      }
    }
  }
}


### 3. POST `/eda/groupby`

Agrégation par groupe avec métriques personnalisables.

Request:
json
{
  "dataset_id": "uuid",
  "by": "segment",
  "metrics": ["mean", "median"]
}


Valeurs valides :
- by : "segment", "channel", "churn"
- metrics : ["mean", "median", "sum", "count"]

### 4. POST `/eda/correlation`

Matrice de corrélation (Pearson) + top paires.

Request:
json
{
  "dataset_id": "uuid"
}


Response (extrait):
json
{
  "result": {
    "matrix": {...},
    "top_pairs": [
      {
        "var1": "income",
        "var2": "spend",
        "correlation": 0.135
      }
    ]
  }
}


### 5. POST /eda/plots

Génère 3 graphiques Plotly JSON.

Request:
json
{
  "dataset_id": "uuid"
}


Graphiques générés (dans artifacts) :
- `histogram_income` : Distribution de income
- `boxplot_income_by_segment` : Income par segment (A/B/C)
- `barplot_segment` : Distribution des segments

##  Exemple d'utilisation complète
bash
# 1. Génération d'un dataset
curl -X POST "http://127.0.0.1:8000/eda/dataset-generate" \
  -H "Content-Type: application/json" \
  -d '{"phase": "eda", "seed": 42, "n": 100}'
# → Notez le dataset_id

# 2. Statistiques descriptives
curl -X POST "http://127.0.0.1:8000/eda/summary" \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "{dataset_id}"}'

# 3. Agrégation par segment
curl -X POST "http://127.0.0.1:8000/eda/groupby" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "{dataset_id}",
    "by": "segment",
    "metrics": ["mean", "median"]
  }'

# 4. Corrélations
curl -X POST "http://127.0.0.1:8000/eda/correlation" \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "{dataset_id}"}'

# 5. Graphiques
curl -X POST "http://127.0.0.1:8000/eda/plots" \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "{dataset_id}"}'


## Défauts injectés dans les datasets

- Valeurs manquantes : 5-10% dans age, income, spend
- Outliers** : ~2% valeurs extrêmes dans income (mean + 10*std)
- Types cohérents : Pas d'erreurs de type (contrairement au TP1)

## Format des graphiques

Les graphiques sont retournés en **Plotly JSON** dans le champ `artifacts`, permettant :
- Visualisation interactive côté client
- Sauvegarde/export facile
- Aucun stockage d'images côté serveur

## Technologies utilisées

- FastAPI : Framework web asynchrone
- Pandas : Statistiques et agrégations
- NumPy : Calculs numériques
- Plotly : Graphiques interactifs (format JSON)
- Pydantic : Validation des données

## Gestion des valeurs manquantes

- Detection : Comptage précis dans `/summary` (missing_rate par variable)
- Robustesse : Les statistiques ignorent automatiquement les NA (pandas)
- Agrégations : Les groupby gèrent correctement les valeurs manquantes

## Licence

Projet académique - TP2