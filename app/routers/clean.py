from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.clean_service import (
    generate_clean_dataset,
    analyze_dataset_quality,
    fit_cleaner,
    transform_with_cleaner
)
import numpy as np
import pandas as pd

router = APIRouter(prefix="/clean", tags=["Clean"])

# -------------------------------
# Pydantic Request Schema
# -------------------------------
class DatasetGenerateRequest(BaseModel):
    phase: str = "clean"
    seed: int
    n: int

class CleanFitRequest(BaseModel):
    dataset_id: str
    impute_strategy: str = "mean"      # mean | median
    outlier_strategy: str = "clip"     # clip | remove
    categorical_strategy: str = "one_hot"  # one_hot | ordinal

class CleanTransformRequest(BaseModel):
    cleaner_id: str

# -------------------------------
# Endpoint: Génération dataset
# -------------------------------
@router.post("/dataset-generate")
def dataset_generate(body: DatasetGenerateRequest):
    try:
        dataset_id, df = generate_clean_dataset(seed=body.seed, n=body.n)

        # Conversion sûre du DataFrame pour JSON
        df_safe = df.copy()
        
        # Remplacer NaN, inf, -inf par None
        df_safe = df_safe.replace([np.nan, np.inf, -np.inf], None)

        sample = df_safe.head(20).to_dict(orient="records")

        response = {
            "meta": {
                "datasetid": dataset_id,
                "phase": body.phase,
                "n_rows": len(df_safe),
                "schema_version": 1,
            },
            "result": {
                "columns": list(df_safe.columns),
                "datasample": sample,
            },
            "report": {},
            "artifacts": {},
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------
# Endpoint: Report
# -------------------------------
@router.get("/report/{dataset_id}")
def clean_report(dataset_id: str):
    """
    Analyse la qualité d'un dataset sans le transformer.
    Retourne un rapport détaillé sur les problèmes de qualité.
    """
    try:
        report = analyze_dataset_quality(dataset_id)
        
        response = {
            "meta": {
                "dataset_id": dataset_id,
                "analysis_type": "quality_report"
            },
            "result": {
                "quality_score": {
                    "missing_rate": round((report["missing_values"]["total"] / 
                                          (report["statistics"]["total_rows"] * 
                                           report["statistics"]["total_columns"])) * 100, 2),
                    "duplicate_rate": report["duplicates"]["percentage"],
                    "has_type_errors": len(report["type_inconsistencies"]) > 0
                }
            },
            "report": report,
            "artifacts": {},
        }
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------
# Endpoint: Fit (IMPLÉMENTÉ)
# -------------------------------
@router.post("/fit")
def clean_fit(body: CleanFitRequest):
    """
    Apprend un pipeline de nettoyage basé sur le dataset et les stratégies choisies.
    Retourne un cleaner_id et un rapport avant nettoyage.
    """
    try:
        cleaner_id, report_before = fit_cleaner(
            dataset_id=body.dataset_id,
            impute_strategy=body.impute_strategy,
            outlier_strategy=body.outlier_strategy,
            categorical_strategy=body.categorical_strategy
        )
        
        response = {
            "meta": {
                "cleaner_id": cleaner_id,
                "dataset_id": body.dataset_id
            },
            "result": {
                "rules": {
                    "impute_strategy": body.impute_strategy,
                    "outlier_strategy": body.outlier_strategy,
                    "categorical_strategy": body.categorical_strategy
                }
            },
            "report": {
                "quality_before": report_before
            },
            "artifacts": {},
        }
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------
# Endpoint: Transform (IMPLÉMENTÉ)
# -------------------------------
@router.post("/transform")
def clean_transform(body: CleanTransformRequest):
    """
    Applique le pipeline de nettoyage et retourne le dataset nettoyé.
    """
    try:
        processed_dataset_id, df_clean, counters = transform_with_cleaner(body.cleaner_id)
        
        # Conversion sûre pour JSON
        df_safe = df_clean.replace([np.nan, np.inf, -np.inf], None)
        sample = df_safe.head(20).to_dict(orient="records")
        
        response = {
            "meta": {
                "processed_dataset_id": processed_dataset_id,
                "cleaner_id": body.cleaner_id
            },
            "result": {
                "rows_before": counters["rows_before"],
                "rows_after": counters["rows_after"],
                "imputations": counters["imputations"],
                "duplicates_removed": counters["duplicates_removed"],
                "outliers_handled": counters["outliers_handled"],
                "type_errors_fixed": counters["type_errors_fixed"],
                "columns": list(df_safe.columns),
                "datasample": sample
            },
            "report": {
                "transformation_summary": counters
            },
            "artifacts": {},
        }
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))