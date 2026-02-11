import numpy as np
import pandas as pd
from uuid import uuid4
from typing import Dict, Any, List
from datetime import datetime
import joblib
from pathlib import Path

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# Import du service ML TP4
from app.services.ml_service import DATASETS_STORAGE

# Stockages
MODELS_STORAGE_ML_Avance: Dict[str, Dict[str, Any]] = {}
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def tune_model(dataset_id: str, model_type: str, search: str, cv: int = 3, seed: int = 42) -> Dict[str, Any]:
    """Optimisation hyperparamètres avec CV"""
    if dataset_id not in DATASETS_STORAGE:
        raise ValueError(f"Dataset {dataset_id} not found")
    
    if model_type not in ["logreg", "rf"]:
        raise ValueError("model_type must be 'logreg' or 'rf'")
    
    if search not in ["grid", "random"]:
        raise ValueError("search must be 'grid' or 'random'")
    
    df = DATASETS_STORAGE[dataset_id].copy()
    
    # Preprocessing (même que TP4)
    X = df.drop("target", axis=1)
    y = df["target"]
    
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    
    numeric_cols = [f"x{i+1}" for i in range(6)]
    imputer = SimpleImputer(strategy="mean")
    X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
    X_valid[numeric_cols] = imputer.transform(X_valid[numeric_cols])
    
    label_encoder = LabelEncoder()
    X_train["segment_encoded"] = label_encoder.fit_transform(X_train["segment"])
    X_valid["segment_encoded"] = label_encoder.transform(X_valid["segment"])
    
    scaler = StandardScaler()
    feature_cols = numeric_cols + ["segment_encoded"]
    X_train_scaled = scaler.fit_transform(X_train[feature_cols])
    X_valid_scaled = scaler.transform(X_valid[feature_cols])
    
    # Grilles de recherche
    if model_type == "logreg":
        base_model = LogisticRegression(max_iter=1000, random_state=seed)
        param_grid = {
            "C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "penalty": ["l2"],
            "solver": ["lbfgs", "liblinear"]
        }
    else:  # rf
        base_model = RandomForestClassifier(random_state=seed)
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    
    # Recherche
    if search == "grid":
        searcher = GridSearchCV(
            base_model, param_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=0
        )
    else:  # random
        searcher = RandomizedSearchCV(
            base_model, param_grid, cv=cv, scoring="f1", n_iter=10, 
            n_jobs=-1, random_state=seed, verbose=0
        )
    
    # Fit
    searcher.fit(X_train_scaled, y_train)
    
    # Meilleur modèle
    best_model = searcher.best_estimator_
    
    # CV results top 5
    cv_results = pd.DataFrame(searcher.cv_results_)
    cv_results = cv_results.sort_values("rank_test_score")
    top_5 = cv_results.head(5)[["params", "mean_test_score", "std_test_score"]].to_dict(orient="records")
    
    # Métriques sur valid
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    y_valid_pred = best_model.predict(X_valid_scaled)
    y_valid_proba = best_model.predict_proba(X_valid_scaled)[:, 1]
    
    valid_metrics = {
        "accuracy": float(accuracy_score(y_valid, y_valid_pred)),
        "precision": float(precision_score(y_valid, y_valid_pred, zero_division=0)),
        "recall": float(recall_score(y_valid, y_valid_pred, zero_division=0)),
        "f1": float(f1_score(y_valid, y_valid_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_valid, y_valid_proba))
    }
    
    # Sauvegarder
    model_id = str(uuid4())
    model_path = MODELS_DIR / f"{model_id}_tuned.joblib"
    
    model_bundle = {
        "model": best_model,
        "imputer": imputer,
        "label_encoder": label_encoder,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "feature_names": feature_cols  # Pour explicabilité
    }
    
    joblib.dump(model_bundle, model_path)
    
    # Métadonnées
    MODELS_STORAGE_ML_Avance[model_id] = {
        "model_id": model_id,
        "model_type": model_type,
        "dataset_id": dataset_id,
        "search_method": search,
        "cv_folds": cv,
        "best_params": searcher.best_params_,
        "best_score_cv": float(searcher.best_score_),
        "valid_metrics": valid_metrics,
        "feature_names": feature_cols,
        "created_at": datetime.now().isoformat(),
        "model_path": str(model_path)
    }
    
    return {
        "best_model_id": model_id,
        "best_params": searcher.best_params_,
        "best_score_cv": float(searcher.best_score_),
        "valid_metrics": valid_metrics,
        "cv_results_summary": top_5,
        "n_configs_tested": len(cv_results)
    }


def get_feature_importance(model_id: str) -> Dict[str, Any]:
    """Importance des features (native)"""
    if model_id not in MODELS_STORAGE_ML_Avance:
        raise ValueError(f"Model {model_id} not found")
    
    model_path = MODELS_STORAGE_ML_Avance[model_id]["model_path"]
    model_bundle = joblib.load(model_path)
    
    model = model_bundle["model"]
    feature_names = model_bundle["feature_names"]
    model_type = MODELS_STORAGE_ML_Avance[model_id]["model_type"]
    
    if model_type == "rf":
        # Importance native RF
        importances = model.feature_importances_
    else:  # logreg
        # Coefficients (valeur absolue pour importance)
        importances = np.abs(model.coef_[0])
    
    # Créer DataFrame et trier
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)
    
    top_features = importance_df.to_dict(orient="records")
    
    return {
        "model_id": model_id,
        "model_type": model_type,
        "method": "native_importance" if model_type == "rf" else "coefficients_abs",
        "feature_importance": top_features,
        "top_5_features": top_features[:5]
    }


def get_permutation_importance(
    model_id: str, dataset_id: str, n_repeats: int = 10, seed: int = 42
) -> Dict[str, Any]:
    """Importance par permutation (agnostique)"""
    if model_id not in MODELS_STORAGE_ML_Avance:
        raise ValueError(f"Model {model_id} not found")
    
    if dataset_id not in DATASETS_STORAGE:
        raise ValueError(f"Dataset {dataset_id} not found")
    
    df = DATASETS_STORAGE[dataset_id].copy()
    X = df.drop("target", axis=1)
    y = df["target"]
    
    # Preprocessing
    model_path = MODELS_STORAGE_ML_Avance[model_id]["model_path"]
    model_bundle = joblib.load(model_path)
    
    model = model_bundle["model"]
    imputer = model_bundle["imputer"]
    label_encoder = model_bundle["label_encoder"]
    scaler = model_bundle["scaler"]
    numeric_cols = model_bundle["numeric_cols"]
    feature_cols = model_bundle["feature_cols"]
    
    X[numeric_cols] = imputer.transform(X[numeric_cols])
    X["segment_encoded"] = label_encoder.transform(X["segment"])
    X_scaled = scaler.transform(X[feature_cols])
    
    # Permutation importance
    perm_importance = permutation_importance(
        model, X_scaled, y, n_repeats=n_repeats, random_state=seed, scoring="f1"
    )
    
    # Résultats
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": perm_importance.importances_mean,
        "importance_std": perm_importance.importances_std
    }).sort_values("importance_mean", ascending=False)
    
    results = importance_df.to_dict(orient="records")
    
    return {
        "model_id": model_id,
        "method": "permutation_importance",
        "n_repeats": n_repeats,
        "feature_importance": results,
        "top_5_features": results[:5]
    }


def explain_instance(model_id: str, instance: Dict[str, Any]) -> Dict[str, Any]:
    """Explication locale d'une prédiction"""
    if model_id not in MODELS_STORAGE_ML_Avance:
        raise ValueError(f"Model {model_id} not found")
    
    # Charger modèle
    model_path = MODELS_STORAGE_ML_Avance[model_id]["model_path"]
    model_bundle = joblib.load(model_path)
    
    model = model_bundle["model"]
    imputer = model_bundle["imputer"]
    label_encoder = model_bundle["label_encoder"]
    scaler = model_bundle["scaler"]
    numeric_cols = model_bundle["numeric_cols"]
    feature_cols = model_bundle["feature_cols"]
    model_type = MODELS_STORAGE_ML_Avance[model_id]["model_type"]
    
    # Preprocessing instance
    df = pd.DataFrame([instance])
    df[numeric_cols] = imputer.transform(df[numeric_cols])
    df["segment_encoded"] = label_encoder.transform(df["segment"])
    X_scaled = scaler.transform(df[feature_cols])
    
    # Prédiction
    prediction = int(model.predict(X_scaled)[0])
    probability = float(model.predict_proba(X_scaled)[0, 1])
    
    # Explication
    if model_type == "logreg":
        # Contribution = coefficient * valeur_standardisée
        coefficients = model.coef_[0]
        intercept = model.intercept_[0]
        
        contributions = []
        for i, feature in enumerate(feature_cols):
            contrib = coefficients[i] * X_scaled[0, i]
            contributions.append({
                "feature": feature,
                "value_scaled": float(X_scaled[0, i]),
                "coefficient": float(coefficients[i]),
                "contribution": float(contrib),
                "pushes_toward": "class_1" if contrib > 0 else "class_0"
            })
        
        # Trier par contribution absolue
        contributions = sorted(contributions, key=lambda x: abs(x["contribution"]), reverse=True)
        
        explanation = {
            "method": "linear_contribution",
            "intercept": float(intercept),
            "contributions": contributions,
            "top_5_factors": contributions[:5]
        }
        
    else:  # rf
        # Approximation via moyenne des arbres (simplifié)
        # Pour RF, on pourrait faire SHAP, mais ici approximation simple
        importances = model.feature_importances_
        
        contributions = []
        for i, feature in enumerate(feature_cols):
            # Contribution approximative = importance * valeur
            contrib = importances[i] * X_scaled[0, i]
            contributions.append({
                "feature": feature,
                "value_scaled": float(X_scaled[0, i]),
                "importance": float(importances[i]),
                "contribution_approx": float(contrib)
            })
        
        contributions = sorted(contributions, key=lambda x: abs(x["contribution_approx"]), reverse=True)
        
        explanation = {
            "method": "importance_weighted",
            "note": "Approximation simple (pas SHAP)",
            "contributions": contributions,
            "top_5_factors": contributions[:5]
        }
    
    return {
        "model_id": model_id,
        "prediction": prediction,
        "probability_class_1": probability,
        "explanation": explanation
    }