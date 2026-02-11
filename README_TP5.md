# TP5 - API ML Avancé : Tuning & Explicabilité

API REST FastAPI pour l'optimisation d'hyperparamètres (GridSearch/RandomSearch + CV) et l'explicabilité des modèles ML (importance features, explication locale).

## Objectifs

Passer de "ça marche" à "c'est défendable" :
- Optimisation propre : GridSearchCV / RandomizedSearchCV avec validation croisée
- Explicabilité globale : Feature importance (native + permutation)
- Explicabilité locale : Contribution de chaque feature à une prédiction

## Fonctionnalités

- Tuning automatisé : Grid Search ou Random Search avec CV 3/5 folds
- Feature importance : Native (RF/LogReg coefficients) + permutation
- Explication locale : Contribution de chaque variable à une prédiction
- Top 5 facteurs : Quelles variables poussent vers classe 0 ou classe 1
- Reproductibilité : Seed obligatoire pour tous les processus stochastiques

## Installation

### Prérequis
- Python 3.12+
- Dataset du TP4 (ou générer un nouveau avec phase="ml")

### Installation des dépendances
bash
pip install fastapi uvicorn pandas numpy scikit-learn joblib

## Démarrage
bash
uvicorn app.main:app --reload


L'API sera accessible sur : `http://127.0.0.1:8000`

Documentation interactive : `http://127.0.0.1:8000/docs`

##  Structure du projet

TP5_API_ML_Avance/
├── app/
│   ├── main.py                    # Point d'entrée FastAPI
│   ├── routers/
│   │   ├── ml.py                  # TP4 - Baseline
│   │   └── ml2.py (ml_avance.py)  # TP5 - Avancé
│   └── services/
│       ├── ml_service.py          # TP4 - Logique baseline
│       └── ml2_service.py         # TP5 - Logique tuning/explainability
├── models/                        # Modèles sérialisés
│   ├── {model_id}.joblib          # TP4 baseline
│   └── {model_id}_tuned.joblib    # TP5 optimisés
├── README_TP5.md
└── TP5_RAPPORT.adoc

## Endpoints

### 1. POST `/ml_avance/tune`

Optimisation d'hyperparamètres avec validation croisée.

Request:
json
{
  "dataset_id": "66963bfa-f590-4ebc-9ae3-47e52037f126",
  "model_type": "logreg",
  "search": "grid",
  "cv": 3,
  "seed": 42
}

Paramètres :
- dataset_id : Dataset du TP4 (réutilisé)
- model_type : "logreg" (Logistic Regression) ou "rf" (Random Forest)
- search : "grid" (exhaustif) ou "random" (échantillonnage)
- cv : 3 ou 5 (nombre de folds)
- seed : Reproductibilité

Grilles de recherche :

Logistic Regression :
python
{
    "C": [0.01, 0.1, 1.0, 10.0, 100.0],
    "penalty": ["l2"],
    "solver": ["lbfgs", "liblinear"]
}
# → 10 configurations testées (5 C × 2 solvers)

Random Forest :
python
{
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}
# → 144 configurations possibles
# Random Search : 10 configurations aléatoires


Response (structure) :
json
{
  "result": {
    "best_model_id": "3af47d62-c39a-4f96-b2c3-bd4a15624d9c",
    "best_params": {
      "C": 10,
      "penalty": "l2",
      "solver": "lbfgs"
    },
    "best_score_cv": 0.514,
    "valid_metrics": {
      "accuracy": 0.733,
      "precision": 0.571,
      "recall": 0.444,
      "f1": 0.500,
      "auc": 0.698
    },
    "cv_results_summary": [
      {
        "params": {"C": 10, "penalty": "l2", "solver": "lbfgs"},
        "mean_test_score": 0.514,
        "std_test_score": 0.236
      }
    ],
    "n_configs_tested": 10
  }
}

Métriques retournées :
- best_score_cv : Meilleur F1-score en validation croisée (moyenne sur k folds)
- valid_metrics : Performance sur validation set (30% des données)
- cv_results_summary** : Top 5 configurations avec scores ± std

### 2. GET /ml_avance/feature-importance/{model_id}

Importance native des features.

Response (structure) :
json
{
  "result": {
    "model_id": "3af47d62-c39a-4f96-b2c3-bd4a15624d9c",
    "model_type": "logreg",
    "method": "coefficients_abs",
    "feature_importance": [
      {"feature": "x1", "importance": 1.852},
      {"feature": "x4", "importance": 0.820},
      {"feature": "x2", "importance": 0.781},
      {"feature": "x6", "importance": 0.430},
      {"feature": "x3", "importance": 0.190},
      {"feature": "segment_encoded", "importance": 0.097},
      {"feature": "x5", "importance": 0.083}
    ],
    "top_5_features": [...]
  }
}

Méthodes selon le type de modèle :
- Logistic Regression : Valeur absolue des coefficients (|β|)
- Random Forest : Importance native (Gini importance)

Interprétation :
- Plus l'importance est élevée, plus la feature influence la prédiction
- Pour LogReg : Coefficient élevé = forte influence sur le score linéaire
- Pour RF : Importance élevée = souvent utilisée pour séparer les classes

### 3. POST `/ml_avance/permutation-importance`

Importance par permutation (méthode agnostique au modèle).

Request:
json
{
  "model_id": "3af47d62-c39a-4f96-b2c3-bd4a15624d9c",
  "dataset_id": "66963bfa-f590-4ebc-9ae3-47e52037f126",
  "n_repeats": 10,
  "seed": 42
}

Paramètres :
- `n_repeats` : Nombre de permutations (plus élevé = plus stable)
- `seed` : Reproductibilité

Principe :
1. Évaluer performance baseline sur dataset complet
2. Pour chaque feature :
   - Permuter aléatoirement les valeurs
   - Réévaluer performance
   - Importance = drop de performance
3. Répéter n_repeats fois pour stabilité

Response (structure) :
json
{
  "result": {
    "method": "permutation_importance",
    "n_repeats": 10,
    "feature_importance": [
      {
        "feature": "x1",
        "importance_mean": 0.280,
        "importance_std": 0.054
      },
      {
        "feature": "x2",
        "importance_mean": 0.083,
        "importance_std": 0.040
      }
    ],
    "top_5_features": [...]
  }
}

Avantages :
- Méthode agnostique (fonctionne pour tout modèle)
- Reflète l'importance réelle dans les prédictions
- Détecte interactions entre features

### 4. POST `/ml_avance/explain-instance`

Explication locale d'une prédiction individuelle.

Request:
json
{
  "model_id": "3af47d62-c39a-4f96-b2c3-bd4a15624d9c",
  "instance": {
    "x1": 0.5,
    "x2": -0.3,
    "x3": 1.2,
    "x4": 0.1,
    "x5": -0.5,
    "x6": 0.8,
    "segment": "A"
  }
}

Response (structure - Logistic Regression) :
json
{
  "result": {
    "prediction": 0,
    "probability_class_1": 0.323,
    "explanation": {
      "method": "linear_contribution",
      "intercept": -1.484,
      "contributions": [
        {
          "feature": "x1",
          "value_scaled": 0.453,
          "coefficient": 1.852,
          "contribution": 0.838,
          "pushes_toward": "class_1"
        },
        {
          "feature": "x6",
          "value_scaled": 0.972,
          "coefficient": 0.430,
          "contribution": 0.418,
          "pushes_toward": "class_1"
        },
        {
          "feature": "x3",
          "value_scaled": 1.345,
          "coefficient": -0.190,
          "contribution": -0.255,
          "pushes_toward": "class_0"
        }
      ],
      "top_5_factors": [...]
    }
  }
}

Calcul de contribution (LogReg) :

Score linéaire = intercept + Σ(coefficient[i] × value_scaled[i])

Contribution[i] = coefficient[i] × value_scaled[i]

Si contribution > 0 → pousse vers classe 1
Si contribution < 0 → pousse vers classe 0

Exemple d'interprétation :

Instance : x1=0.5, x6=0.8, x3=1.2, segment=A

x1 : +0.838 → FORTE poussée vers classe 1 (valeur positive + coef positif)
x6 : +0.418 → MOYENNE poussée vers classe 1
x3 : -0.255 → Pousse vers classe 0 (valeur haute mais coef négatif)

Total : -1.484 + 0.838 + 0.418 - 0.255 - ... = -0.67
→ Score négatif → Prédiction classe 0

Response (structure - Random Forest) :
json
{
  "explanation": {
    "method": "importance_weighted",
    "note": "Approximation simple (pas SHAP)",
    "contributions": [
      {
        "feature": "x1",
        "value_scaled": 0.453,
        "importance": 0.280,
        "contribution_approx": 0.127
      }
    ]
  }
}

## Exemple d'utilisation complète
bash
# 1. PRÉREQUIS : Dataset TP4
# (Utilisez dataset_id d'un dataset déjà généré en TP4)

# 2. Tuning avec GridSearch
curl -X POST "http://127.0.0.1:8000/ml_avance/tune" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "66963bfa-f590-4ebc-9ae3-47e52037f126",
    "model_type": "logreg",
    "search": "grid",
    "cv": 3,
    "seed": 42
  }'
# → best_model_id: "3af47d62-c39a-4f96-b2c3-bd4a15624d9c"

# 3. Feature importance native
curl -X GET "http://127.0.0.1:8000/ml_avance/feature-importance/3af47d62-c39a-4f96-b2c3-bd4a15624d9c"

# 4. Permutation importance
curl -X POST "http://127.0.0.1:8000/ml_avance/permutation-importance" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "3af47d62-c39a-4f96-b2c3-bd4a15624d9c",
    "dataset_id": "66963bfa-f590-4ebc-9ae3-47e52037f126",
    "n_repeats": 10,
    "seed": 42
  }'

# 5. Explication locale
curl -X POST "http://127.0.0.1:8000/ml_avance/explain-instance" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "3af47d62-c39a-4f96-b2c3-bd4a15624d9c",
    "instance": {
      "x1": 0.5, "x2": -0.3, "x3": 1.2,
      "x4": 0.1, "x5": -0.5, "x6": 0.8,
      "segment": "A"
    }
  }'

## Comparaison TP4 vs TP5

### TP4 - Baseline (sans tuning)

Model: LogisticRegression(max_iter=1000, C=1.0)
Valid F1: 0.500


### TP5 - Après tuning

Model: LogisticRegression(max_iter=1000, C=10.0)
Best CV F1: 0.514
Valid F1: 0.500

Observation : Le tuning a exploré 10 configurations. Le meilleur modèle (C=10) performe légèrement mieux en CV mais similaire en validation. Cela suggère que le baseline était déjà proche de l'optimum.

## Explicabilité : Deux niveaux

### Niveau 1 : Global (Feature Importance)

Question : Quelles variables sont importantes en général ?

Réponse (exemple réel) :

1. x1 : 1.85  → Variable la plus importante
2. x4 : 0.82
3. x2 : 0.78
4. x6 : 0.43
5. x3 : 0.19

Utilité :
- Sélection de features
- Compréhension du modèle
- Communication aux métiers

### Niveau 2 : Local (Explain Instance)

Question :Pourquoi cette observation a été classée ainsi ?

Réponse (exemple réel) :

Instance: x1=0.5, x6=0.8, segment=A
Prédiction: Classe 0

Facteurs POUR classe 1:
- x1 : +0.84 (valeur positive + coefficient élevé)
- x6 : +0.42

Facteurs CONTRE classe 1:
- x3 : -0.26 (valeur élevée mais coefficient négatif)
- x2 : -0.10
- segment : -0.09

Bilan: -1.48 + 0.84 + 0.42 - 0.45 = -0.67 → Classe 0

Utilité :
- Débogage modèle
- Conformité réglementaire (RGPD)
- Confiance utilisateurs

## Technologies utilisées

- FastAPI : Framework web
- Scikit-learn : GridSearchCV, RandomizedSearchCV, permutation_importance
- Joblib : Sérialisation modèles optimisés
- Pandas/NumPy : Manipulation données

## Reproductibilité

Tous les processus stochastiques sont contrôlés par seed :
- Train/valid split (seed)
- GridSearch/RandomSearch (seed dans RandomizedSearchCV)
- Permutation importance (seed)

Garantie : Mêmes résultats à chaque exécution avec même seed.

## Licence

Projet académique - TP5