import numpy as np
import pandas as pd
from uuid import uuid4
from typing import Dict, Any, List, Tuple
from datetime import datetime
import joblib
import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)

# Stockages
DATASETS_STORAGE: Dict[str, pd.DataFrame] = {}
MODELS_STORAGE: Dict[str, Dict[str, Any]] = {}

# Créer dossier models si inexistant
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def generate_ml_dataset(seed: int, n: int):
    """Génère dataset pour ML avec classification binaire déséquilibrée"""
    rng = np.random.default_rng(seed)
    
    # Variables numériques x1..x6
    X = rng.normal(loc=0, scale=1, size=(n, 6))
    df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(6)])
    
    # Variable catégorielle segment (A/B/C)
    segment = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
    df["segment"] = segment
    
    # Target binaire avec déséquilibre 70/30
    # Créer une combinaison linéaire pour avoir une relation avec X
    logit = (
        0.5 * df["x1"] + 
        0.3 * df["x2"] - 
        0.4 * df["x3"] + 
        0.2 * (df["segment"] == "A").astype(int) -
        0.3 * (df["segment"] == "C").astype(int)
    )
    
    # Ajouter du bruit
    noise = rng.normal(0, 1, size=n)
    prob = 1 / (1 + np.exp(-(logit + noise)))
    
    # Ajuster pour avoir ~30% de classe 1
    threshold = np.percentile(prob, 70)
    target = (prob > threshold).astype(int)
    
    df["target"] = target
    
    # Inject NA (2-5%)
    for col in [f"x{i+1}" for i in range(6)]:
        mask = rng.random(n) < rng.uniform(0.02, 0.05)
        df.loc[mask, col] = np.nan
    
    dataset_id = str(uuid4())
    DATASETS_STORAGE[dataset_id] = df.copy()
    
    df_display = df.replace([np.nan, np.inf, -np.inf], None)
    return dataset_id, df_display


def train_model(dataset_id: str, model_type: str, test_size: float = 0.3, seed: int = 42) -> Dict[str, Any]:
    """Entraîne un modèle ML avec preprocessing"""
    if dataset_id not in DATASETS_STORAGE:
        raise ValueError(f"Dataset {dataset_id} not found")
    
    if model_type not in ["logreg", "rf"]:
        raise ValueError("model_type must be 'logreg' or 'rf'")
    
    df = DATASETS_STORAGE[dataset_id].copy()
    
    # Séparer features et target
    X = df.drop("target", axis=1)
    y = df["target"]
    
    # Split train/valid
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    # Preprocessing
    # 1. Imputation numériques
    numeric_cols = [f"x{i+1}" for i in range(6)]
    imputer = SimpleImputer(strategy="mean")
    X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
    X_valid[numeric_cols] = imputer.transform(X_valid[numeric_cols])
    
    # 2. Encoding catégoriel
    label_encoder = LabelEncoder()
    X_train["segment_encoded"] = label_encoder.fit_transform(X_train["segment"])
    X_valid["segment_encoded"] = label_encoder.transform(X_valid["segment"])
    
    # 3. Standardisation
    scaler = StandardScaler()
    feature_cols = numeric_cols + ["segment_encoded"]
    X_train_scaled = scaler.fit_transform(X_train[feature_cols])
    X_valid_scaled = scaler.transform(X_valid[feature_cols])
    
    # Entraînement
    if model_type == "logreg":
        model = LogisticRegression(max_iter=1000, random_state=seed)
        hyperparams = {"max_iter": 1000, "random_state": seed, "solver": "lbfgs"}
    else:  # rf
        model = RandomForestClassifier(n_estimators=100, random_state=seed, max_depth=10)
        hyperparams = {"n_estimators": 100, "random_state": seed, "max_depth": 10}
    
    model.fit(X_train_scaled, y_train)
    
    # Prédictions
    y_train_pred = model.predict(X_train_scaled)
    y_valid_pred = model.predict(X_valid_scaled)
    
    # Probabilités
    y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
    y_valid_proba = model.predict_proba(X_valid_scaled)[:, 1]
    
    # Métriques train
    train_metrics = {
        "accuracy": float(accuracy_score(y_train, y_train_pred)),
        "precision": float(precision_score(y_train, y_train_pred, zero_division=0)),
        "recall": float(recall_score(y_train, y_train_pred, zero_division=0)),
        "f1": float(f1_score(y_train, y_train_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_train, y_train_proba))
    }
    
    # Métriques valid
    valid_metrics = {
        "accuracy": float(accuracy_score(y_valid, y_valid_pred)),
        "precision": float(precision_score(y_valid, y_valid_pred, zero_division=0)),
        "recall": float(recall_score(y_valid, y_valid_pred, zero_division=0)),
        "f1": float(f1_score(y_valid, y_valid_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_valid, y_valid_proba))
    }
    
    # Sauvegarder modèle
    model_id = str(uuid4())
    model_path = MODELS_DIR / f"{model_id}.joblib"
    
    model_bundle = {
        "model": model,
        "imputer": imputer,
        "label_encoder": label_encoder,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols
    }
    
    joblib.dump(model_bundle, model_path)
    
    # Stocker métadonnées
    MODELS_STORAGE[model_id] = {
        "model_id": model_id,
        "model_type": model_type,
        "dataset_id": dataset_id,
        "hyperparams": hyperparams,
        "features_used": feature_cols,
        "train_size": len(X_train),
        "valid_size": len(X_valid),
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "created_at": datetime.now().isoformat(),
        "model_path": str(model_path),
        "preprocessing": {
            "imputation": "mean",
            "encoding": "LabelEncoder (segment → int)",
            "scaling": "StandardScaler"
        }
    }
    
    return {
        "model_id": model_id,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "features_used": feature_cols,
        "train_size": len(X_train),
        "valid_size": len(X_valid)
    }


def get_metrics(model_id: str) -> Dict[str, Any]:
    """Récupère les métriques d'un modèle"""
    if model_id not in MODELS_STORAGE:
        raise ValueError(f"Model {model_id} not found")
    
    info = MODELS_STORAGE[model_id]
    
    return {
        "model_id": model_id,
        "train_metrics": info["train_metrics"],
        "valid_metrics": info["valid_metrics"],
        "interpretation": {
            "best_metric": "f1" if info["valid_metrics"]["f1"] > 0.7 else "accuracy",
            "overfitting": "Yes" if (info["train_metrics"]["accuracy"] - info["valid_metrics"]["accuracy"]) > 0.1 else "No"
        }
    }


def predict(model_id: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Prédit sur nouvelles données"""
    if model_id not in MODELS_STORAGE:
        raise ValueError(f"Model {model_id} not found")
    
    # Charger modèle
    model_path = MODELS_STORAGE[model_id]["model_path"]
    model_bundle = joblib.load(model_path)
    
    model = model_bundle["model"]
    imputer = model_bundle["imputer"]
    label_encoder = model_bundle["label_encoder"]
    scaler = model_bundle["scaler"]
    numeric_cols = model_bundle["numeric_cols"]
    
    # Créer DataFrame
    df = pd.DataFrame(data)
    
    # Vérifier colonnes requises
    required_cols = numeric_cols + ["segment"]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    # Preprocessing (même pipeline)
    df[numeric_cols] = imputer.transform(df[numeric_cols])
    df["segment_encoded"] = label_encoder.transform(df["segment"])
    
    feature_cols = numeric_cols + ["segment_encoded"]
    X_scaled = scaler.transform(df[feature_cols])
    
    # Prédictions
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    
    return {
        "predictions": [int(p) for p in predictions],
        "probabilities": [float(p) for p in probabilities],
        "n_samples": len(predictions)
    }


def get_model_info(model_id: str) -> Dict[str, Any]:
    """Récupère info complète d'un modèle"""
    if model_id not in MODELS_STORAGE:
        raise ValueError(f"Model {model_id} not found")
    
    info = MODELS_STORAGE[model_id]
    
    return {
        "model_id": model_id,
        "model_type": info["model_type"],
        "dataset_id": info["dataset_id"],
        "hyperparams": info["hyperparams"],
        "features_used": info["features_used"],
        "preprocessing": info["preprocessing"],
        "train_size": info["train_size"],
        "valid_size": info["valid_size"],
        "created_at": info["created_at"],
        "model_path": info["model_path"]
    }