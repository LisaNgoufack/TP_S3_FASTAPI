# TP4 - API ML Baseline : Entraînement, Métriques & Prédiction

API REST FastAPI pour l'entraînement de modèles de Machine Learning supervisés (classification binaire) avec pipeline de preprocessing, métriques complètes et prédictions.

## Objectifs

Construire une API qui :
- Entraîne des modèles baseline (Logistic Regression, Random Forest)
- Expose les métriques de performance (accuracy, precision, recall, f1, AUC)
- Prédit sur de nouvelles données avec probabilités
- Sérialise les modèles pour réutilisation
- Documente preprocessing et hyperparamètres

## Fonctionnalités

- Génération de datasets ML : Classification binaire déséquilibrée (70/30)
- Entraînement supervisé : Logistic Regression et Random Forest
- Pipeline preprocessing : Imputation, encoding catégoriel, standardisation
- Métriques complètes : Train/Valid split avec 5 métriques
- Prédictions : Classes + probabilités sur nouvelles données
- Sérialisation : Sauvegarde modèles + preprocessing pipeline

##  Installation

### Prérequis
- Python 3.12+
- pip

### Installation des dépendances
bash
pip install fastapi uvicorn pandas numpy pydantic scikit-learn joblib


## Démarrage
bash
uvicorn app.main:app --reload


L'API sera accessible sur : `http://127.0.0.1:8000`

Documentation interactive : `http://127.0.0.1:8000/docs`

##  Structure du projet

TP4_API_ML/
├── app/
│   ├── main.py                 # Point d'entrée FastAPI
│   ├── routers/
│   │   └── ml.py               # Endpoints ML
│   └── services/
│       └── ml_service.py       # Logique métier ML
├── models/                     # Modèles sérialisés (.joblib)
│   └── {model_id}.joblib
├── README_TP4.md
└── TP4_RAPPORT.adoc


## Endpoints

### 1. POST /ml/dataset-generate

Génère un dataset pour classification binaire.

Request:
json
{
  "phase": "ml",
  "seed": 42,
  "n": 200
}

Dataset généré :
- Variables numériques : x1, x2, x3, x4, x5, x6
- Variable catégorielle : segment (A/B/C)
- Cible binaire : target (0/1)
- Déséquilibre : ~70% classe 0, ~30% classe 1
- NA : 2-5% par variable numérique
- Relation : Target dépend linéairement de x1-x3 + segment

Response (extrait) :
json
{
  "meta": {
    "datasetid": "uuid",
    "phase": "ml",
    "n_rows": 200,
    "target_distribution": {
      "class_0": 140,
      "class_1": 60
    }
  }
}

### 2. POST /ml/train

Entraîne un modèle ML baseline.

Request:
json
{
  "dataset_id": "uuid",
  "model_type": "logreg",
  "test_size": 0.3,
  "seed": 42
}

Paramètres :
- model_type : "logreg" (Logistic Regression) ou "rf" (Random Forest)
- test_size : Proportion train/valid (0.0 à 1.0)
- seed : Reproductibilité du split

Response (structure) :
json
{
  "result": {
    "model_id": "867a3d2f-ec9f-4688-a378-c8edb04ccec9",
    "train_metrics": {
      "accuracy": 0.80,
      "precision": 0.73,
      "recall": 0.52,
      "f1": 0.61,
      "auc": 0.87
    },
    "valid_metrics": {
      "accuracy": 0.73,
      "precision": 0.57,
      "recall": 0.44,
      "f1": 0.50,
      "auc": 0.69
    },
    "features_used": ["x1", "x2", "x3", "x4", "x5", "x6", "segment_encoded"],
    "train_size": 70,
    "valid_size": 30
  }
}

Preprocessing automatique :
1. Imputation : Mean pour variables numériques (NA → moyenne)
2. Encoding : LabelEncoder pour segment (A→0, B→1, C→2)
3. Scaling : StandardScaler (μ=0, σ=1)

Modèles disponibles :
- logreg : Logistic Regression (max_iter=1000, solver=lbfgs)
- rf : Random Forest (n_estimators=100, max_depth=10)

### 3. GET /ml/metrics/{model_id}

Récupère les métriques d'un modèle entraîné.

Response (structure) :
json
{
  "result": {
    "model_id": "867a3d2f...",
    "train_metrics": {
      "accuracy": 0.80,
      "precision": 0.73,
      "recall": 0.52,
      "f1": 0.61,
      "auc": 0.87
    },
    "valid_metrics": {
      "accuracy": 0.73,
      "precision": 0.57,
      "recall": 0.44,
      "f1": 0.50,
      "auc": 0.69
    },
    "interpretation": {
      "best_metric": "accuracy",
      "overfitting": "No"
    }
  }
}

Métriques fournies :
- Accuracy : % prédictions correctes
- Precision : % vrais positifs parmi prédits positifs
- Recall : % vrais positifs détectés
- F1-Score : Moyenne harmonique precision/recall
- AUC-ROC : Aire sous courbe ROC (discrimination classes)

Interprétation automatique :
- best_metric : Métrique recommandée selon performance
- overfitting : Détection si écart train/valid > 10%


### 4. POST /ml/predict

Prédit sur nouvelles données.

Request:
json
{
  "model_id": "867a3d2f-ec9f-4688-a378-c8edb04ccec9",
  "data": [
    {
      "x1": 0.5,
      "x2": -0.3,
      "x3": 1.2,
      "x4": 0.1,
      "x5": -0.5,
      "x6": 0.8,
      "segment": "A"
    }
  ]
}

Colonnes requises :
- x1, x2, x3, x4, x5, x6 (numériques)
- segment (A, B, ou C)

Response (structure) :
json
{
  "result": {
    "predictions": [0],
    "probabilities": [0.323],
    "n_samples": 1
  }
}

Interprétation :
- predictions[i] : Classe prédite (0 ou 1)
-`probabilities[i] : P(classe 1) - probabilité classe positive

Exemple :

Prédiction = 0
Probabilité = 0.323

→ Modèle prédit classe 0 avec 67.7% de confiance
  (car P(classe 0) = 1 - 0.323 = 0.677)


### 5. GET /ml/model-info/{model_id}

Info complète sur un modèle entraîné.

Response (structure) :
json
{
  "result": {
    "model_id": "867a3d2f...",
    "model_type": "logreg",
    "dataset_id": "7ed737dd...",
    "hyperparams": {
      "max_iter": 1000,
      "random_state": 42,
      "solver": "lbfgs"
    },
    "features_used": ["x1", "x2", ..., "segment_encoded"],
    "preprocessing": {
      "imputation": "mean",
      "encoding": "LabelEncoder (segment → int)",
      "scaling": "StandardScaler"
    },
    "train_size": 70,
    "valid_size": 30,
    "created_at": "2026-02-11T10:17:06.150234",
    "model_path": "models/867a3d2f-ec9f-4688-a378-c8edb04ccec9.joblib"
  }
}

Informations retournées :
- Hyperparamètres : Configuration exacte du modèle
- Features : Variables utilisées (incluant encodage)
- Preprocessing : Pipeline appliqué (reproductible)
- Timestamps : Date de création
- Path : Chemin du fichier sérialisé

## Exemple d'utilisation complète
bash
# 1. Générer dataset ML
curl -X POST "http://127.0.0.1:8000/ml/dataset-generate" \
  -H "Content-Type: application/json" \
  -d '{"phase": "ml", "seed": 42, "n": 200}'
# → dataset_id: 7ed737dd-fc32-408e-8489-e0512c5d5df0

# 2. Entraîner Logistic Regression
curl -X POST "http://127.0.0.1:8000/ml/train" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "7ed737dd-fc32-408e-8489-e0512c5d5df0",
    "model_type": "logreg",
    "test_size": 0.3,
    "seed": 42
  }'
# → model_id: 867a3d2f-ec9f-4688-a378-c8edb04ccec9

# 3. Récupérer métriques
curl -X GET "http://127.0.0.1:8000/ml/metrics/867a3d2f-ec9f-4688-a378-c8edb04ccec9"

# 4. Prédire sur nouvelles données
curl -X POST "http://127.0.0.1:8000/ml/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "867a3d2f-ec9f-4688-a378-c8edb04ccec9",
    "data": [
      {"x1": 0.5, "x2": -0.3, "x3": 1.2, "x4": 0.1, "x5": -0.5, "x6": 0.8, "segment": "A"}
    ]
  }'

# 5. Info modèle
curl -X GET "http://127.0.0.1:8000/ml/model-info/867a3d2f-ec9f-4688-a378-c8edb04ccec9"

## Pipeline ML complet

1. Génération dataset
   ↓
2. Split train/valid (stratifié)
   ↓
3. Preprocessing
   ├─ Imputation NA (mean)
   ├─ Encoding catégoriel (LabelEncoder)
   └─ Scaling (StandardScaler)
   ↓
4. Entraînement modèle
   ↓
5. Évaluation (5 métriques)
   ↓
6. Sérialisation (joblib)
   ↓
7. Prédictions nouvelles données


### Preprocessing détaillé

Avant preprocessing :

x1: [0.5, NaN, 1.2, ...]
x2: [-0.3, 0.8, NaN, ...]
segment: ['A', 'B', 'C', ...]

Après preprocessing :

x1: [0.5, 0.7, 1.2, ...]        # NaN → mean
x2: [-0.3, 0.8, 0.2, ...]       # NaN → mean
segment_encoded: [0, 1, 2, ...] # A→0, B→1, C→2

Puis standardisation (μ=0, σ=1) :
x1_scaled: [-0.2, 0.1, 0.8, ...]


## Sérialisation des modèles

### Format de sauvegarde

Chaque modèle est sauvegardé en joblib avec :
- Modèle entraîné (sklearn)
- Imputer (SimpleImputer)
- LabelEncoder (catégorie → int)
- Scaler (StandardScaler)
- Métadonnées (features, columns)

Fichier : models/{model_id}.joblib

### Chargement pour prédiction
python
# L'API charge automatiquement :
model_bundle = joblib.load(f"models/{model_id}.joblib")

# Contient :
- model_bundle["model"]          # LogisticRegression ou RandomForest
- model_bundle["imputer"]        # SimpleImputer
- model_bundle["label_encoder"]  # LabelEncoder
- model_bundle["scaler"]         # StandardScaler
- model_bundle["feature_cols"]   # Liste features

Avantage : Pipeline complet reproductible sans configuration manuelle.


## Technologies utilisées

- FastAPI : Framework web asynchrone
- Pandas : Manipulation de données
- NumPy : Calculs numériques
- Scikit-learn : ML (LogisticRegression, RandomForest, metrics)
- Joblib : Sérialisation efficace des modèles
- Pydantic : Validation des schémas


## Gestion des valeurs manquantes

### Stratégie d'imputation

Dans le dataset :
- 2-5% de NA par variable numérique
- NA injectés aléatoirement

Preprocessing :
- Méthode : Imputation par moyenne (SimpleImputer strategy="mean")
- Application : Sur x1-x6 uniquement (numériques)
- Justification : Méthode baseline simple et efficace

Exemple :

x1 avant : [0.5, NaN, 1.2, 0.8, NaN]
x1 mean  : (0.5 + 1.2 + 0.8) / 3 = 0.83
x1 après : [0.5, 0.83, 1.2, 0.8, 0.83]


## Interprétation des métriques

### Accuracy (Exactitude)
Formule : (TP + TN) / Total

Interprétation :
- 0.80 → 80% des prédictions sont correctes
- Limite : Trompeuse si classes déséquilibrées

### Precision (Précision)
Formule : TP / (TP + FP)

Interprétation :
- 0.73 → Parmi les prédictions "classe 1", 73% sont vraies
- Important si coût des faux positifs élevé

### Recall (Rappel / Sensibilité)
Formule : TP / (TP + FN)

Interprétation :
- 0.52 → 52% des vrais "classe 1" sont détectés
- Important si coût des faux négatifs élevé

### F1-Score
Formule : 2 × (Precision × Recall) / (Precision + Recall)

Interprétation :
- 0.61 → Équilibre entre precision et recall
- Utile pour classes déséquilibrées

### AUC-ROC
Formule : Aire sous courbe ROC

Interprétation :
- 0.87 → Excellente capacité de discrimination
- 1.0 = Parfait, 0.5 = Aléatoire

## Licence

Projet académique - TP4