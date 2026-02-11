# TP3 - API Analyse Multivariée : PCA & Clustering

API REST FastAPI pour l'analyse multivariée avec réduction de dimensionnalité (PCA) et clustering (K-means).

## Objectifs

Exposer des méthodes multivariées via API avec résultats interprétables :
- PCA : Réduction de dimensionnalité avec projection, variance expliquée et loadings
- K-means : Clustering avec labels, centroids et qualité (silhouette)
- Rapport : Interprétation automatique des composantes principales et suggestions de clustering

## Fonctionnalités

- Génération de datasets : 8 variables avec 3 clusters simulés + colinéarité
- PCA : Projection PC1..PCk, variance expliquée, loadings par variable
- K-means : Labels, centroids, silhouette score, inertie
- Rapport interprétable : Top variables sur PC1/PC2, tailles clusters, suggestions

## Installation

### Prérequis
- Python 3.12+
- pip

### Installation des dépendances
bash
pip install fastapi uvicorn pandas numpy pydantic scikit-learn

## Démarrage
bash
uvicorn app.main:app --reload


L'API sera accessible sur : `http://127.0.0.1:8000`

Documentation interactive : `http://127.0.0.1:8000/docs`

## Structure du projet

TP3_API_MV/
├── app/
│   ├── main.py                 # Point d'entrée FastAPI
│   ├── routers/
│   │   └── mv.py               # Endpoints analyse multivariée
│   └── services/
│       └── mv_service.py       # Logique métier PCA/Clustering
├── README_TP3.md
└── TP3_RAPPORT.adoc


## Endpoints

### 1. POST `/mv/dataset-generate`

Génère un dataset pour analyse multivariée.

Request:
json
{
  "phase": "mv",
  "seed": 42,
  "n": 100
}


Dataset généré :
- 8 variables numériques : x1, x2, x3, x4, x5, x6, x7, x8
- Structure : 3 clusters simulés (centres différents)
- Colinéarité : x5 ≈ x1 + bruit
- NA : 2-5% par variable


### 2. POST `/mv/pca/fit_transform`

Analyse en composantes principales (PCA).

Request:
json
{
  "dataset_id": "uuid",
  "n_components": 2,
  "scale": true
}


**Paramètres :**
- n_components : 2 à 5 (nombre de composantes)
- scale : true/false (standardisation Z-score)

Response (structure) :
json
{
  "result": {
    "projection": [
      {"PC1": -0.804, "PC2": 0.045},
      {"PC1": -0.633, "PC2": -0.151}
    ],
    "explained_variance_ratio": {
      "PC1": 0.933,
      "PC2": 0.020
    },
    "total_variance_explained": 0.953,
    "loadings": {
      "PC1": {
        "x1": 0.356, "x2": 0.352, ...
      },
      "PC2": {
        "x1": 0.570, "x2": -0.163, ...
      }
    },
    "n_samples": 83,
    "n_components": 2
  },
  "report": {
    "interpretation": "Les 2 premières composantes expliquent 95.3% de la variance totale"
  }
}


Interprétation des résultats :
- projection : Coordonnées de chaque observation dans l'espace réduit
- explained_variance_ratio : % de variance capturée par chaque PC
- loadings : Contribution de chaque variable à chaque composante
- total_variance_explained : Variance totale capturée


### 3. POST `/mv/cluster/kmeans`

Clustering K-means avec métriques de qualité.

Request:
json
{
  "dataset_id": "uuid",
  "k": 3,
  "scale": true
}


Paramètres :
- k : 2 à 6 (nombre de clusters)
- scale : true/false (standardisation)

Response (structure) :
json
{
  "result": {
    "labels": [2, 2, 0, 0, 1, ...],
    "centroids": [
      {"x1": 4.71, "x2": 4.92, ...},
      {"x1": -2.95, "x2": -2.61, ...},
      {"x1": 0.08, "x2": -0.01, ...}
    ],
    "cluster_sizes": {
      "cluster_0": 30,
      "cluster_1": 25,
      "cluster_2": 28
    },
    "silhouette_score": 0.630,
    "inertia": 56.96,
    "n_samples": 83,
    "k": 3
  },
  "report": {
    "cluster_sizes": {...},
    "quality": "Silhouette score: 0.630"
  }
}


Métriques de qualité :
- Silhouette score : -1 à 1 (proche de 1 = bon clustering)
- Inertia : Somme distances intra-cluster (plus faible = mieux)


### 4. GET `/mv/report/{dataset_id}`

Rapport interprétable automatique.

Response (structure) :
json
{
  "result": {
    "pca_interpretation": {
      "PC1_top_positive": {"x4": 0.356, "x1": 0.356, "x8": 0.355},
      "PC1_top_negative": {"x3": 0.350, "x6": 0.351, "x2": 0.352},
      "PC2_top_positive": {"x5": 0.576, "x1": 0.570},
      "PC2_top_negative": {"x3": -0.501, "x4": -0.173},
      "variance_explained": {
        "PC1": 0.933,
        "PC2": 0.020,
        "total_2PC": 0.953
      }
    },
    "clustering_suggestion": {
      "inertias_by_k": {
        "k=2": 133.54,
        "k=3": 56.96,
        "k=4": 49.89,
        "k=5": 46.60
      },
      "note": "Plus faible inertie = meilleur ajustement (mais attention overfitting)"
    },
    "dataset_info": {
      "n_samples": 83,
      "n_features": 8,
      "missing_rows_removed": 17
    }
  }
}


Utilité du rapport :
- Identifie les variables les plus importantes sur PC1/PC2
- Suggère le nombre optimal de clusters (via courbe d'inertie)
- Donne un aperçu de la qualité des données


## Exemple d'utilisation complète
bash
# 1. Générer dataset
curl -X POST "http://127.0.0.1:8000/mv/dataset-generate" \
  -H "Content-Type: application/json" \
  -d '{"phase": "mv", "seed": 42, "n": 150}'
# → Notez le dataset_id

# 2. PCA avec 2 composantes
curl -X POST "http://127.0.0.1:8000/mv/pca/fit_transform" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "{dataset_id}",
    "n_components": 2,
    "scale": true
  }'

# 3. K-means avec 3 clusters
curl -X POST "http://127.0.0.1:8000/mv/cluster/kmeans" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "{dataset_id}",
    "k": 3,
    "scale": true
  }'

# 4. Rapport interprétable
curl -X GET "http://127.0.0.1:8000/mv/report/{dataset_id}"


## Caractéristiques du dataset

### Structure des clusters simulés

Le dataset génère 3 clusters avec des centres distincts :
- Cluster 1 : Centre autour de (0, 0, 0, ...)
- Cluster 2 : Centre autour de (5, 5, 5, ...)
- Cluster 3 : Centre autour de (-3, -3, -3, ...)

### Colinéarité

- x5 ≈ x1 + bruit : Permet de tester la détection de redondance via PCA
- Les loadings de PC1 devraient être similaires pour x1 et x5

### Valeurs manquantes

- 2-5% de NA par variable (aléatoire)
- Gestion : Suppression des lignes avec NA avant analyse



## Interprétation des résultats

### PCA : Comment interpréter

1. Variance expliquée : Si PC1+PC2 > 80%, les 2 composantes capturent l'essentiel
2. Loadings : Variables avec loading élevé (>0.4) contribuent fortement à la PC
3. Projection : Points proches dans l'espace réduit sont similaires

Exemple :

PC1 explique 93% → Première composante = axe principal de variation
x1, x4, x8 ont loadings élevés sur PC1 → Ces variables définissent PC1


### K-means : Comment interpréter

1. Silhouette > 0.5 : Bon clustering
2. Silhouette < 0.3 : Clustering faible, revoir k
3. Inertia décroissante : Plus k augmente, meilleure est l'inertie (attention overfitting)

Stratégie :
- Tester k=2 à k=6
- Chercher le "coude" dans la courbe d'inertie (elbow method)
- Valider avec silhouette score


## Technologies utilisées

- FastAPI : Framework web asynchrone
- Pandas : Manipulation de données
- NumPy : Calculs matriciels
- Scikit-learn : PCA, K-means, StandardScaler, Silhouette
- Pydantic : Validation des schémas


## Gestion des valeurs manquantes

- Stratégie : Suppression des lignes avec NA avant PCA/Clustering
- Justification : PCA et K-means nécessitent des données complètes
- Alternative possible : Imputation (non implémentée dans ce TP)

## Licence

Projet académique - TP3