import numpy as np
import pandas as pd
from uuid import uuid4
from typing import Dict, Any, List
import plotly.express as px

# Stockage datasets
DATASETS_STORAGE: Dict[str, pd.DataFrame] = {}

def generate_eda_dataset(seed: int, n: int):
    """Génère dataset pour EDA"""
    rng = np.random.default_rng(seed)
    
    # Variables numériques
    age = rng.integers(18, 70, size=n)
    income = rng.normal(50000, 15000, size=n)
    spend = rng.normal(2000, 800, size=n)
    visits = rng.integers(1, 50, size=n)
    
    # Variables catégorielles
    segment = rng.choice(["A", "B", "C"], size=n, p=[0.4, 0.35, 0.25])
    channel = rng.choice(["web", "store", "app"], size=n, p=[0.5, 0.3, 0.2])
    churn = rng.integers(0, 2, size=n)
    
    df = pd.DataFrame({
        "age": age,
        "income": income,
        "spend": spend,
        "visits": visits,
        "segment": segment,
        "channel": channel,
        "churn": churn
    })
    
    # Inject NA (5-10%)
    for col in ["age", "income", "spend"]:
        mask = rng.random(n) < rng.uniform(0.05, 0.10)
        df.loc[mask, col] = np.nan
    
    # Inject outliers in income
    n_out = max(1, int(n * 0.02))
    out_idx = rng.integers(0, n, size=n_out)
    df.loc[out_idx, "income"] = df["income"].mean() + 10 * df["income"].std()
    
    dataset_id = str(uuid4())
    DATASETS_STORAGE[dataset_id] = df.copy()
    
    df = df.replace([np.nan, np.inf, -np.inf], None)
    return dataset_id, df


def compute_summary(dataset_id: str) -> Dict[str, Any]:
    """Statistiques descriptives par variable"""
    if dataset_id not in DATASETS_STORAGE:
        raise ValueError(f"Dataset {dataset_id} not found")
    
    df = DATASETS_STORAGE[dataset_id]
    summary = {}
    
    # Numériques
    numeric_cols = ["age", "income", "spend", "visits"]
    for col in numeric_cols:
        stats = df[col].describe()
        summary[col] = {
            "count": int(df[col].count()),
            "mean": float(stats["mean"]) if not pd.isna(stats["mean"]) else None,
            "std": float(stats["std"]) if not pd.isna(stats["std"]) else None,
            "min": float(stats["min"]) if not pd.isna(stats["min"]) else None,
            "25%": float(stats["25%"]) if not pd.isna(stats["25%"]) else None,
            "50%": float(stats["50%"]) if not pd.isna(stats["50%"]) else None,
            "75%": float(stats["75%"]) if not pd.isna(stats["75%"]) else None,
            "max": float(stats["max"]) if not pd.isna(stats["max"]) else None,
            "missing_rate": float((df[col].isna().sum() / len(df)) * 100)
        }
    
    # Catégorielles
    cat_cols = ["segment", "channel", "churn"]
    for col in cat_cols:
        value_counts = df[col].value_counts().to_dict()
        summary[col] = {
            "unique": int(df[col].nunique()),
            "top": str(df[col].mode()[0]) if len(df[col].mode()) > 0 else None,
            "freq": int(df[col].value_counts().max()) if len(df[col]) > 0 else 0,
            "distribution": {str(k): int(v) for k, v in value_counts.items()}
        }
    
    return summary


def compute_groupby(dataset_id: str, by: str, metrics: List[str]) -> List[Dict]:
    """Agrégation par groupe"""
    if dataset_id not in DATASETS_STORAGE:
        raise ValueError(f"Dataset {dataset_id} not found")
    
    df = DATASETS_STORAGE[dataset_id]
    numeric_cols = ["age", "income", "spend", "visits"]
    
    # Groupby
    grouped = df.groupby(by)[numeric_cols]
    
    result = []
    for metric in metrics:
        if metric == "mean":
            agg = grouped.mean()
        elif metric == "median":
            agg = grouped.median()
        elif metric == "sum":
            agg = grouped.sum()
        elif metric == "count":
            agg = grouped.count()
        else:
            continue
        
        agg_dict = agg.reset_index().to_dict(orient="records")
        result.append({"metric": metric, "data": agg_dict})
    
    return result


def compute_correlation(dataset_id: str) -> Dict[str, Any]:
    """Matrice de corrélation"""
    if dataset_id not in DATASETS_STORAGE:
        raise ValueError(f"Dataset {dataset_id} not found")
    
    df = DATASETS_STORAGE[dataset_id]
    numeric_cols = ["age", "income", "spend", "visits"]
    
    corr_matrix = df[numeric_cols].corr()
    
    # Top paires
    corr_pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            corr_pairs.append({
                "var1": numeric_cols[i],
                "var2": numeric_cols[j],
                "correlation": float(corr_matrix.iloc[i, j])
            })
    
    corr_pairs = sorted(corr_pairs, key=lambda x: abs(x["correlation"]), reverse=True)
    
    return {
        "matrix": corr_matrix.to_dict(),
        "top_pairs": corr_pairs[:5]
    }


def generate_plots(dataset_id: str) -> Dict[str, Any]:
    """Génère graphiques Plotly JSON"""
    if dataset_id not in DATASETS_STORAGE:
        raise ValueError(f"Dataset {dataset_id} not found")
    
    df = DATASETS_STORAGE[dataset_id]
    
    # 1. Histogramme income
    fig1 = px.histogram(df, x="income", nbins=30, title="Distribution Income")
    hist_json = fig1.to_json()
    
    # 2. Boxplot income par segment
    fig2 = px.box(df, x="segment", y="income", title="Income by Segment")
    box_json = fig2.to_json()
    
    # 3. Barplot distribution segment
    segment_counts = df["segment"].value_counts().reset_index()
    segment_counts.columns = ["segment", "count"]
    fig3 = px.bar(segment_counts, x="segment", y="count", title="Segment Distribution")
    bar_json = fig3.to_json()
    
    return {
        "histogram_income": hist_json,
        "boxplot_income_by_segment": box_json,
        "barplot_segment": bar_json
    }