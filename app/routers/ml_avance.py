from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from app.services.ml_avance_service import (
    tune_model,
    get_feature_importance,
    get_permutation_importance,
    explain_instance
)

router = APIRouter(prefix="/ml_avance", tags=["ML Advanced"])


class TuneRequest(BaseModel):
    dataset_id: str
    model_type: str  # "logreg" ou "rf"
    search: str  # "grid" ou "random"
    cv: int = 3
    seed: int = 42


class PermutationImportanceRequest(BaseModel):
    model_id: str
    dataset_id: str
    n_repeats: int = 10
    seed: int = 42


class ExplainInstanceRequest(BaseModel):
    model_id: str
    instance: Dict[str, Any]


@router.post("/tune")
def tune_endpoint(body: TuneRequest):
    """Optimisation hyperparamètres avec CV"""
    try:
        if body.model_type not in ["logreg", "rf"]:
            raise ValueError("model_type must be 'logreg' or 'rf'")
        
        if body.search not in ["grid", "random"]:
            raise ValueError("search must be 'grid' or 'random'")
        
        if body.cv not in [3, 5]:
            raise ValueError("cv must be 3 or 5")
        
        result = tune_model(
            dataset_id=body.dataset_id,
            model_type=body.model_type,
            search=body.search,
            cv=body.cv,
            seed=body.seed
        )
        
        return {
            "meta": {
                "dataset_id": body.dataset_id,
                "model_type": body.model_type,
                "search": body.search,
                "cv": body.cv
            },
            "result": result,
            "report": {
                "message": f"Best CV F1: {result['best_score_cv']:.3f}, Valid F1: {result['valid_metrics']['f1']:.3f}"
            },
            "artifacts": {}
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feature-importance/{model_id}")
def feature_importance_endpoint(model_id: str):
    """Importance des features (native)"""
    try:
        importance = get_feature_importance(model_id)
        
        return {
            "meta": {"model_id": model_id},
            "result": importance,
            "report": {},
            "artifacts": {}
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/permutation-importance")
def permutation_importance_endpoint(body: PermutationImportanceRequest):
    """Importance par permutation (agnostique)"""
    try:
        importance = get_permutation_importance(
            model_id=body.model_id,
            dataset_id=body.dataset_id,
            n_repeats=body.n_repeats,
            seed=body.seed
        )
        
        return {
            "meta": {"model_id": body.model_id, "n_repeats": body.n_repeats},
            "result": importance,
            "report": {},
            "artifacts": {}
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explain-instance")
def explain_instance_endpoint(body: ExplainInstanceRequest):
    """Explication locale d'une prédiction"""
    try:
        explanation = explain_instance(body.model_id, body.instance)
        
        return {
            "meta": {"model_id": body.model_id},
            "result": explanation,
            "report": {},
            "artifacts": {}
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))