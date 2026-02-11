from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from app.services.mv_service import (
    generate_mv_dataset,
    pca_fit_transform,
    kmeans_clustering,
    generate_mv_report
)

router = APIRouter(prefix="/mv", tags=["Multivariate Analysis"])

class DatasetGenerateRequest(BaseModel):
    phase: str = "mv"
    seed: int
    n: int

class PCARequest(BaseModel):
    dataset_id: str
    n_components: int = 2  # 2 à 5
    scale: bool = True

class KMeansRequest(BaseModel):
    dataset_id: str
    k: int = 3  # 2 à 6
    scale: bool = True


@router.post("/dataset-generate")
def dataset_generate(body: DatasetGenerateRequest):
    """Génère dataset pour analyse multivariée"""
    try:
        dataset_id, df = generate_mv_dataset(seed=body.seed, n=body.n)
        sample = df.head(20).to_dict(orient="records")
        
        return {
            "meta": {"datasetid": dataset_id, "phase": body.phase, "n_rows": len(df)},
            "result": {"columns": list(df.columns), "datasample": sample},
            "report": {},
            "artifacts": {}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pca/fit_transform")
def pca_endpoint(body: PCARequest):
    """PCA : projection, variance expliquée, loadings"""
    try:
        if body.n_components < 2 or body.n_components > 5:
            raise ValueError("n_components must be between 2 and 5")
        
        result = pca_fit_transform(body.dataset_id, body.n_components, body.scale)
        
        return {
            "meta": {
                "dataset_id": body.dataset_id,
                "method": "PCA",
                "n_components": body.n_components,
                "scaled": body.scale
            },
            "result": result,
            "report": {
                "interpretation": f"Les {body.n_components} premières composantes expliquent {result['total_variance_explained']*100:.1f}% de la variance totale"
            },
            "artifacts": {}
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cluster/kmeans")
def kmeans_endpoint(body: KMeansRequest):
    """K-means clustering avec silhouette score"""
    try:
        if body.k < 2 or body.k > 6:
            raise ValueError("k must be between 2 and 6")
        
        result = kmeans_clustering(body.dataset_id, body.k, body.scale)
        
        return {
            "meta": {
                "dataset_id": body.dataset_id,
                "method": "K-Means",
                "k": body.k,
                "scaled": body.scale
            },
            "result": result,
            "report": {
                "cluster_sizes": result["cluster_sizes"],
                "quality": f"Silhouette score: {result['silhouette_score']:.3f}" if result["silhouette_score"] else "N/A"
            },
            "artifacts": {}
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report/{dataset_id}")
def mv_report(dataset_id: str):
    """Rapport interprétable : top loadings + suggestion clustering"""
    try:
        report = generate_mv_report(dataset_id)
        
        return {
            "meta": {"dataset_id": dataset_id, "report_type": "multivariate_analysis"},
            "result": report,
            "report": {},
            "artifacts": {}
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))