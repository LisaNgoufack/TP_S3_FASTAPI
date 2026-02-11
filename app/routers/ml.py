from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from app.services.ml_service import (
    generate_ml_dataset,
    train_model,
    get_metrics,
    predict,
    get_model_info
)

router = APIRouter(prefix="/ml", tags=["Machine Learning"])


class DatasetGenerateRequest(BaseModel):
    phase: str = "ml"
    seed: int
    n: int


class TrainRequest(BaseModel):
    dataset_id: str
    model_type: str  # "logreg" ou "rf"
    test_size: float = 0.3
    seed: int = 42


class PredictRequest(BaseModel):
    model_id: str
    data: List[Dict[str, Any]]


@router.post("/dataset-generate")
def dataset_generate(body: DatasetGenerateRequest):
    """Génère dataset ML pour classification binaire"""
    try:
        dataset_id, df = generate_ml_dataset(seed=body.seed, n=body.n)
        sample = df.head(20).to_dict(orient="records")
        
        # Stats target
        target_counts = df["target"].value_counts().to_dict()
        
        return {
            "meta": {
                "datasetid": dataset_id,
                "phase": body.phase,
                "n_rows": len(df),
                "target_distribution": {
                    "class_0": int(target_counts.get(0, 0)),
                    "class_1": int(target_counts.get(1, 0))
                }
            },
            "result": {"columns": list(df.columns), "datasample": sample},
            "report": {},
            "artifacts": {}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train")
def train_endpoint(body: TrainRequest):
    """Entraîne modèle ML baseline"""
    try:
        if body.model_type not in ["logreg", "rf"]:
            raise ValueError("model_type must be 'logreg' or 'rf'")
        
        result = train_model(
            dataset_id=body.dataset_id,
            model_type=body.model_type,
            test_size=body.test_size,
            seed=body.seed
        )
        
        return {
            "meta": {
                "dataset_id": body.dataset_id,
                "model_type": body.model_type,
                "test_size": body.test_size
            },
            "result": result,
            "report": {
                "message": f"Model trained successfully. Valid F1: {result['valid_metrics']['f1']:.3f}"
            },
            "artifacts": {}
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{model_id}")
def metrics_endpoint(model_id: str):
    """Récupère métriques d'un modèle"""
    try:
        metrics = get_metrics(model_id)
        
        return {
            "meta": {"model_id": model_id},
            "result": metrics,
            "report": {},
            "artifacts": {}
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict")
def predict_endpoint(body: PredictRequest):
    """Prédictions sur nouvelles données"""
    try:
        predictions = predict(body.model_id, body.data)
        
        return {
            "meta": {"model_id": body.model_id, "n_samples": len(body.data)},
            "result": predictions,
            "report": {},
            "artifacts": {}
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-info/{model_id}")
def model_info_endpoint(model_id: str):
    """Info complète sur un modèle"""
    try:
        info = get_model_info(model_id)
        
        return {
            "meta": {"model_id": model_id},
            "result": info,
            "report": {},
            "artifacts": {}
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))