import numpy as np
import pandas as pd
from uuid import uuid4
from typing import Dict, Any, Tuple

# Stockage en mémoire des datasets générés
DATASETS_STORAGE: Dict[str, pd.DataFrame] = {}

# Stockage en mémoire des pipelines de nettoyage
CLEANERS_STORAGE: Dict[str, Dict[str, Any]] = {}


def generate_clean_dataset(seed: int, n: int):
    rng = np.random.default_rng(seed)

    # Variables de base
    x1 = rng.normal(loc=0, scale=1, size=n)
    x2 = rng.normal(loc=5, scale=2, size=n)
    x3 = rng.normal(loc=10, scale=3, size=n)

    segments = rng.choice(["A", "B", "C"], size=n, p=[0.4, 0.4, 0.2])
    target = rng.integers(0, 2, size=n)

    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "segment": segments, "target": target})

    # --- Missing values (10–20%) ---
    for col in ["x1", "x2", "x3"]:
        mask = rng.random(n) < rng.uniform(0.10, 0.20)
        df.loc[mask, col] = np.nan

    # --- Doublons (~5%) ---
    n_dups = max(1, int(n * 0.05))
    dup_indices = rng.integers(0, n, size=n_dups)
    df = pd.concat([df, df.iloc[dup_indices]], ignore_index=True)

    # --- Outliers (1–3 valeurs extrêmes) ---
    for col in ["x1", "x2", "x3"]:
        n_out = rng.integers(1, 4)
        out_idx = rng.integers(0, len(df), size=n_out)
        df.loc[out_idx, col] = df[col].mean() + 10 * df[col].std()

    # --- Types cassés dans x2 ---
    df["x2"] = df["x2"].astype(object)
    bad_idx = rng.integers(0, len(df), size=3)
    for idx in bad_idx:
        df.at[idx, "x2"] = "oops"

    # ID du dataset
    dataset_id = str(uuid4())
    
    # Sauvegarder le dataset AVANT conversion JSON
    DATASETS_STORAGE[dataset_id] = df.copy()

    # --- Conversion sûre pour API JSON ---
    df = df.replace([np.nan, np.inf, -np.inf], None)

    return dataset_id, df


def analyze_dataset_quality(dataset_id: str) -> Dict[str, Any]:
    """
    Analyse la qualité d'un dataset sans le transformer.
    Retourne un rapport détaillé sur les problèmes de qualité.
    """
    if dataset_id not in DATASETS_STORAGE:
        raise ValueError(f"Dataset {dataset_id} not found")
    
    df = DATASETS_STORAGE[dataset_id]
    
    # --- 1. Analyse des valeurs manquantes ---
    missing_analysis = {}
    for col in df.columns:
        n_missing = df[col].isna().sum()
        missing_rate = (n_missing / len(df)) * 100
        missing_analysis[col] = {
            "count": int(n_missing),
            "percentage": round(missing_rate, 2)
        }
    
    total_missing = df.isna().sum().sum()
    
    # --- 2. Analyse des doublons ---
    n_duplicates = df.duplicated().sum()
    duplicate_rate = (n_duplicates / len(df)) * 100
    
    # --- 3. Analyse des outliers (méthode 3-sigma) ---
    outliers_analysis = {}
    numeric_cols = ["x1", "x2", "x3"]
    
    for col in numeric_cols:
        # Convertir en numérique, forcer les non-numériques à NaN
        numeric_values = pd.to_numeric(df[col], errors='coerce')
        
        mean = numeric_values.mean()
        std = numeric_values.std()
        
        # Outliers = valeurs > mean ± 3*std
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
        
        outliers = numeric_values[(numeric_values < lower_bound) | (numeric_values > upper_bound)]
        n_outliers = len(outliers)
        outlier_rate = (n_outliers / len(numeric_values.dropna())) * 100 if len(numeric_values.dropna()) > 0 else 0
        
        outliers_analysis[col] = {
            "count": int(n_outliers),
            "percentage": round(outlier_rate, 2),
            "bounds": {
                "lower": round(lower_bound, 2),
                "upper": round(upper_bound, 2)
            }
        }
    
    # --- 4. Analyse des types incohérents ---
    type_issues = {}
    for col in numeric_cols:
        # Compter les valeurs non-numériques
        non_numeric = pd.to_numeric(df[col], errors='coerce').isna() & df[col].notna()
        n_type_errors = non_numeric.sum()
        
        if n_type_errors > 0:
            examples = df.loc[non_numeric, col].head(3).tolist()
            type_issues[col] = {
                "count": int(n_type_errors),
                "examples": examples
            }
    
    # --- 5. Statistiques générales ---
    stats = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "numeric_columns": len(numeric_cols),
        "categorical_columns": len(df.columns) - len(numeric_cols)
    }
    
    # --- Construction du rapport ---
    report = {
        "missing_values": {
            "total": int(total_missing),
            "by_column": missing_analysis
        },
        "duplicates": {
            "count": int(n_duplicates),
            "percentage": round(duplicate_rate, 2)
        },
        "outliers": outliers_analysis,
        "type_inconsistencies": type_issues,
        "statistics": stats
    }
    
    return report


def fit_cleaner(
    dataset_id: str,
    impute_strategy: str,
    outlier_strategy: str,
    categorical_strategy: str
) -> Tuple[str, Dict[str, Any]]:
    """
    Apprend un pipeline de nettoyage basé sur le dataset et les stratégies choisies.
    Retourne un cleaner_id et un rapport avant nettoyage.
    """
    if dataset_id not in DATASETS_STORAGE:
        raise ValueError(f"Dataset {dataset_id} not found")
    
    df = DATASETS_STORAGE[dataset_id].copy()
    numeric_cols = ["x1", "x2", "x3"]
    
    # --- Calcul des statistiques pour l'imputation ---
    imputation_values = {}
    for col in numeric_cols:
        numeric_values = pd.to_numeric(df[col], errors='coerce')
        
        if impute_strategy == "mean":
            imputation_values[col] = numeric_values.mean()
        elif impute_strategy == "median":
            imputation_values[col] = numeric_values.median()
        else:
            imputation_values[col] = numeric_values.mean()  # défaut
    
    # --- Calcul des bornes pour les outliers ---
    outlier_bounds = {}
    for col in numeric_cols:
        numeric_values = pd.to_numeric(df[col], errors='coerce')
        mean = numeric_values.mean()
        std = numeric_values.std()
        
        outlier_bounds[col] = {
            "lower": mean - 3 * std,
            "upper": mean + 3 * std
        }
    
    # --- Encodage catégoriel (préparation) ---
    if categorical_strategy == "one_hot":
        unique_segments = df["segment"].unique().tolist()
        categorical_mapping = {"strategy": "one_hot", "categories": unique_segments}
    elif categorical_strategy == "ordinal":
        unique_segments = sorted(df["segment"].unique().tolist())
        categorical_mapping = {
            "strategy": "ordinal",
            "mapping": {cat: idx for idx, cat in enumerate(unique_segments)}
        }
    else:
        categorical_mapping = {"strategy": "none"}
    
    # --- Génération du cleaner_id et sauvegarde du pipeline ---
    cleaner_id = str(uuid4())
    
    pipeline = {
        "dataset_id": dataset_id,
        "impute_strategy": impute_strategy,
        "imputation_values": imputation_values,
        "outlier_strategy": outlier_strategy,
        "outlier_bounds": outlier_bounds,
        "categorical_strategy": categorical_strategy,
        "categorical_mapping": categorical_mapping
    }
    
    CLEANERS_STORAGE[cleaner_id] = pipeline
    
    # --- Génération du rapport AVANT nettoyage ---
    report_before = analyze_dataset_quality(dataset_id)
    
    return cleaner_id, report_before


def transform_with_cleaner(cleaner_id: str) -> Tuple[str, pd.DataFrame, Dict[str, Any]]:
    """
    Applique le pipeline de nettoyage et retourne le dataset nettoyé.
    Retourne : processed_dataset_id, df_clean, compteurs
    """
    if cleaner_id not in CLEANERS_STORAGE:
        raise ValueError(f"Cleaner {cleaner_id} not found")
    
    pipeline = CLEANERS_STORAGE[cleaner_id]
    dataset_id = pipeline["dataset_id"]
    
    if dataset_id not in DATASETS_STORAGE:
        raise ValueError(f"Dataset {dataset_id} not found")
    
    df = DATASETS_STORAGE[dataset_id].copy()
    numeric_cols = ["x1", "x2", "x3"]
    
    # Compteurs
    counters = {
        "rows_before": int(len(df)),  # Conversion en int Python
        "imputations": 0,
        "duplicates_removed": 0,
        "outliers_handled": 0,
        "type_errors_fixed": 0
    }
    
    # --- 1. Correction des types incohérents ---
    for col in numeric_cols:
        non_numeric_mask = pd.to_numeric(df[col], errors='coerce').isna() & df[col].notna()
        n_type_errors = int(non_numeric_mask.sum())  # Conversion en int Python
        
        if n_type_errors > 0:
            counters["type_errors_fixed"] += n_type_errors
            # Convertir en numérique, les valeurs invalides deviennent NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # --- 2. Imputation des valeurs manquantes ---
    imputation_values = pipeline["imputation_values"]
    for col in numeric_cols:
        n_missing = int(df[col].isna().sum())  # Conversion en int Python
        if n_missing > 0:
            counters["imputations"] += n_missing
            # FIX: Utiliser assignment au lieu de inplace
            df[col] = df[col].fillna(imputation_values[col])
    
    # --- 3. Gestion des outliers ---
    outlier_strategy = pipeline["outlier_strategy"]
    outlier_bounds = pipeline["outlier_bounds"]
    
    if outlier_strategy == "clip":
        for col in numeric_cols:
            lower = outlier_bounds[col]["lower"]
            upper = outlier_bounds[col]["upper"]
            
            outliers_mask = (df[col] < lower) | (df[col] > upper)
            n_outliers = int(outliers_mask.sum())  # Conversion en int Python
            counters["outliers_handled"] += n_outliers
            
            df[col] = df[col].clip(lower=lower, upper=upper)
    
    elif outlier_strategy == "remove":
        mask = pd.Series([True] * len(df))
        for col in numeric_cols:
            lower = outlier_bounds[col]["lower"]
            upper = outlier_bounds[col]["upper"]
            
            outliers_mask = (df[col] < lower) | (df[col] > upper)
            counters["outliers_handled"] += int(outliers_mask.sum())  # Conversion en int Python
            
            mask = mask & ~outliers_mask
        
        df = df[mask].reset_index(drop=True)
    
    # --- 4. Suppression des doublons ---
    n_duplicates = int(df.duplicated().sum())  # Conversion en int Python
    counters["duplicates_removed"] = n_duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    
    # --- 5. Encodage des variables catégorielles ---
    categorical_mapping = pipeline["categorical_mapping"]
    
    if categorical_mapping["strategy"] == "one_hot":
        df = pd.get_dummies(df, columns=["segment"], prefix="segment")
    elif categorical_mapping["strategy"] == "ordinal":
        df["segment"] = df["segment"].map(categorical_mapping["mapping"])
    
    counters["rows_after"] = int(len(df))  # Conversion en int Python
    
    # --- Sauvegarde du dataset nettoyé ---
    processed_dataset_id = str(uuid4())
    DATASETS_STORAGE[processed_dataset_id] = df.copy()
    
    return processed_dataset_id, df, counters