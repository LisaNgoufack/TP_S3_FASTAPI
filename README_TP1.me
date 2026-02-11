# TP1 - API Nettoyage & Préparation de Données

API REST FastAPI pour le nettoyage et la préparation de datasets avec valeurs manquantes, doublons, outliers et types incohérents.

## Objectifs

Transformer une table sale en table préparée pour l'analyse/modélisation via un pipeline de nettoyage configurable.

## Fonctionnalités

- Génération de datasets avec défauts injectés (missing, doublons, outliers, types cassés)
- Analyse de qualité sans transformation
- Apprentissage de pipeline de nettoyage
- Application du nettoyage avec traçabilité complète

## Installation

### Prérequis
- Python 3.12+
- pip

### Installation des dépendances
```bash
# Création d'un environnement virtuel
python -m venv .venv

# Activation de  l'environnement virtuel
# Windows
.venv\Scripts\activate

# Installation des dépendances
pip install fastapi uvicorn pandas numpy pydantic
```

## Démarrage
```bash
uvicorn app.main:app --reload
```

L'API sera accessible sur : `http://127.0.0.1:8000`

Documentation interactive : `http://127.0.0.1:8000/docs`

##  Structure du projet

TP1_API_Clean/
├── app/
│   ├── main.py                 # Point d'entrée FastAPI
│   ├── routers/
│   │   └── clean.py            # Endpoints de nettoyage
│   └── services/
│       └── clean_service.py    # Logique métier
├── README.md
└── TP1_RAPPORT.adoc            # Document de rendu
```

##  Endpoints

### 1. POST /clean/dataset-generate

Génère un dataset avec défauts injectés.

**Request:**

json
{
  "phase": "clean",
  "seed": 42,
  "n": 100
}


**Response:**

json
{
  "meta": {
    "datasetid": "uuid",
    "phase": "clean",
    "n_rows": 105
  },
  "result": {
    "columns": ["x1", "x2", "x3", "segment", "target"],
    "datasample": [...]
  }
}


### 2. GET /clean/report/{dataset_id}

Analyse la qualité d'un dataset sans transformation.

**Response:**
json

{
  "report": {
    "missing_values": {...},
    "duplicates": {...},
    "outliers": {...},
    "type_inconsistencies": {...}
  }
}


### 3. POST `/clean/fit`

Apprend un pipeline de nettoyage.

**Request:**
json

{
  "dataset_id": "uuid",
  "impute_strategy": "mean",
  "outlier_strategy": "clip",
  "categorical_strategy": "one_hot"
}


**Response:**
json

{
  "meta": {
    "cleaner_id": "uuid"
  },
  "report": {
    "quality_before": {...}
  }
}


### 4. POST `/clean/transform`

Applique le pipeline de nettoyage.

**Request:**
json

{
  "cleaner_id": "uuid"
}

**Response:**
json

{
  "result": {
    "rows_before": 105,
    "rows_after": 100,
    "imputations": 56,
    "duplicates_removed": 5,
    "outliers_handled": 4,
    "type_errors_fixed": 3
  }
}


##  Exemple d'utilisation complète
```bash

# 1. Génération d'un dataset
curl -X POST "http://127.0.0.1:8000/clean/dataset-generate" \
  -H "Content-Type: application/json" \
  -d '{"phase": "clean", "seed": 42, "n": 100}'
# → Notez le dataset_id

# 2. Analyser la qualité
curl -X GET "http://127.0.0.1:8000/clean/report/{dataset_id}"

# 3. Apprendre le pipeline
curl -X POST "http://127.0.0.1:8000/clean/fit" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "{dataset_id}",
    "impute_strategy": "mean",
    "outlier_strategy": "clip",
    "categorical_strategy": "one_hot"
  }'
# → Notez le cleaner_id

# 4. Appliquer le nettoyage
curl -X POST "http://127.0.0.1:8000/clean/transform" \
  -H "Content-Type: application/json" \
  -d '{"cleaner_id": "{cleaner_id}"}'


##  Configuration

### Stratégies d'imputation
- mean : Remplacement par la moyenne
- median : Remplacement par la médiane

### Stratégies de gestion des outliers
- clip : Limitation aux bornes (mean ± 3*std)
- remove : Suppression des lignes

### Stratégies d'encodage catégoriel
- one_hot : Encodage binaire (segment → segment_A, segment_B, segment_C)
- ordinal : Encodage ordinal (A→0, B→1, C→2)

## Défauts injectés dans les datasets

- Valeurs manquantes : 10-20% par colonne numérique
- Doublons : ~5% de lignes dupliquées
- Outliers : 1-3 valeurs extrêmes par colonne
- Types incohérents : Chaînes "oops" dans colonne numérique x2

## Technologies utilisées

- FastAPI : Framework web asynchrone
- Pandas : Manipulation de données
- NumPy : Calculs numériques
- Pydantic : Validation des données

## 📝 Licence

Projet académique - TP1