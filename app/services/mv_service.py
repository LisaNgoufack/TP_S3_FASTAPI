import numpy as np
import pandas as pd
from uuid import uuid4
from typing import Dict, Any, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Stockage datasets
DATASETS_STORAGE: Dict[str, pd.DataFrame] = {}

def generate_mv_dataset(seed: int, n: int):
    """Génère dataset pour analyse multivariée avec 3 clusters simulés"""
    rng = np.random.default_rng(seed)
    
    # 3 clusters avec centres différents
    n_per_cluster = n // 3
    
    # Cluster 1
    cluster1 = rng.normal(loc=[0, 0, 0, 0, 0, 0, 0, 0], scale=1, size=(n_per_cluster, 8))
    
    # Cluster 2
    cluster2 = rng.normal(loc=[5, 5, 5, 5, 5, 5, 5, 5], scale=1, size=(n_per_cluster, 8))
    
    # Cluster 3
    cluster3 = rng.normal(loc=[-3, -3, -3, -3, -3, -3, -3, -3], scale=1, size=(n_per_cluster, 8))
    
    # Combiner
    data = np.vstack([cluster1, cluster2, cluster3])
    
    # Ajouter des lignes pour atteindre n
    remaining = n - len(data)
    if remaining > 0:
        extra = rng.normal(loc=[2, 2, 2, 2, 2, 2, 2, 2], scale=2, size=(remaining, 8))
        data = np.vstack([data, extra])
    
    df = pd.DataFrame(data, columns=[f"x{i+1}" for i in range(8)])
    
    # Créer colinéarité : x5 ≈ x1 + bruit
    df["x5"] = df["x1"] + rng.normal(0, 0.3, size=len(df))
    
    # Inject NA (2-5%)
    for col in df.columns:
        mask = rng.random(len(df)) < rng.uniform(0.02, 0.05)
        df.loc[mask, col] = np.nan
    
    dataset_id = str(uuid4())
    DATASETS_STORAGE[dataset_id] = df.copy()
    
    df = df.replace([np.nan, np.inf, -np.inf], None)
    return dataset_id, df


def pca_fit_transform(dataset_id: str, n_components: int, scale: bool) -> Dict[str, Any]:
    """PCA avec projection, variance expliquée et loadings"""
    if dataset_id not in DATASETS_STORAGE:
        raise ValueError(f"Dataset {dataset_id} not found")
    
    df = DATASETS_STORAGE[dataset_id].copy()
    
    # Supprimer les NA
    df_clean = df.dropna()
    
    # Standardisation optionnelle
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_clean)
    else:
        X_scaled = df_clean.values
    
    # PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Projection (records)
    projection_df = pd.DataFrame(
        X_pca,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    projection = projection_df.to_dict(orient="records")
    
    # Explained variance
    explained_variance = {
        f"PC{i+1}": float(pca.explained_variance_ratio_[i])
        for i in range(n_components)
    }
    
    # Loadings (contributions des variables)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=df_clean.columns
    )
    
    loadings_dict = {}
    for pc in loadings.columns:
        loadings_dict[pc] = {
            var: float(loadings.loc[var, pc])
            for var in loadings.index
        }
    
    return {
        "projection": projection,
        "explained_variance_ratio": explained_variance,
        "total_variance_explained": float(sum(pca.explained_variance_ratio_)),
        "loadings": loadings_dict,
        "n_samples": len(df_clean),
        "n_components": n_components
    }


def kmeans_clustering(dataset_id: str, k: int, scale: bool) -> Dict[str, Any]:
    """K-means clustering avec labels, centroids et silhouette"""
    if dataset_id not in DATASETS_STORAGE:
        raise ValueError(f"Dataset {dataset_id} not found")
    
    df = DATASETS_STORAGE[dataset_id].copy()
    
    # Supprimer les NA
    df_clean = df.dropna()
    
    # Standardisation optionnelle
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_clean)
    else:
        X_scaled = df_clean.values
    
    # K-means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Silhouette score
    if k > 1 and len(df_clean) > k:
        silhouette = float(silhouette_score(X_scaled, labels))
    else:
        silhouette = None
    
    # Centroids (dans l'espace original si scaled)
    if scale:
        centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    else:
        centroids = kmeans.cluster_centers_
    
    centroids_df = pd.DataFrame(
        centroids,
        columns=df_clean.columns
    )
    centroids_dict = centroids_df.to_dict(orient="records")
    
    # Cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = {f"cluster_{int(u)}": int(c) for u, c in zip(unique, counts)}
    
    return {
        "labels": [int(l) for l in labels],
        "centroids": centroids_dict,
        "cluster_sizes": cluster_sizes,
        "silhouette_score": silhouette,
        "inertia": float(kmeans.inertia_),
        "n_samples": len(df_clean),
        "k": k
    }


def generate_mv_report(dataset_id: str) -> Dict[str, Any]:
    """Rapport interprétable : top loadings PC1/PC2 + tailles clusters suggérées"""
    if dataset_id not in DATASETS_STORAGE:
        raise ValueError(f"Dataset {dataset_id} not found")
    
    df = DATASETS_STORAGE[dataset_id].copy()
    df_clean = df.dropna()
    
    # PCA avec 2 composantes pour interprétation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    
    # Top loadings PC1
    loadings_pc1 = pd.Series(pca.components_[0], index=df_clean.columns)
    top_pc1_pos = loadings_pc1.nlargest(3).to_dict()
    top_pc1_neg = loadings_pc1.nsmallest(3).to_dict()
    
    # Top loadings PC2
    loadings_pc2 = pd.Series(pca.components_[1], index=df_clean.columns)
    top_pc2_pos = loadings_pc2.nlargest(3).to_dict()
    top_pc2_neg = loadings_pc2.nsmallest(3).to_dict()
    
    # Suggestion de K optimal (via inertie pour k=2 à 5)
    inertias = {}
    for k in range(2, 6):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias[f"k={k}"] = float(kmeans.inertia_)
    
    return {
        "pca_interpretation": {
            "PC1_top_positive": {k: float(v) for k, v in top_pc1_pos.items()},
            "PC1_top_negative": {k: float(v) for k, v in top_pc1_neg.items()},
            "PC2_top_positive": {k: float(v) for k, v in top_pc2_pos.items()},
            "PC2_top_negative": {k: float(v) for k, v in top_pc2_neg.items()},
            "variance_explained": {
                "PC1": float(pca.explained_variance_ratio_[0]),
                "PC2": float(pca.explained_variance_ratio_[1]),
                "total_2PC": float(sum(pca.explained_variance_ratio_[:2]))
            }
        },
        "clustering_suggestion": {
            "inertias_by_k": inertias,
            "note": "Plus faible inertie = meilleur ajustement (mais attention overfitting)"
        },
        "dataset_info": {
            "n_samples": len(df_clean),
            "n_features": len(df_clean.columns),
            "missing_rows_removed": len(df) - len(df_clean)
        }
    }