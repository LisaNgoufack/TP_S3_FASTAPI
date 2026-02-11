from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from app.services.eda_service import (
    generate_eda_dataset,
    compute_summary,
    compute_groupby,
    compute_correlation,
    generate_plots
)

router = APIRouter(prefix="/eda", tags=["EDA"])

class DatasetGenerateRequest(BaseModel):
    phase: str = "eda"
    seed: int
    n: int

class SummaryRequest(BaseModel):
    dataset_id: str

class GroupbyRequest(BaseModel):
    dataset_id: str
    by: str
    metrics: List[str]

class CorrelationRequest(BaseModel):
    dataset_id: str

class PlotsRequest(BaseModel):
    dataset_id: str


@router.post("/dataset-generate")
def dataset_generate(body: DatasetGenerateRequest):
    try:
        dataset_id, df = generate_eda_dataset(seed=body.seed, n=body.n)
        sample = df.head(20).to_dict(orient="records")
        
        return {
            "meta": {"datasetid": dataset_id, "phase": body.phase, "n_rows": len(df)},
            "result": {"columns": list(df.columns), "datasample": sample},
            "report": {},
            "artifacts": {}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/summary")
def eda_summary(body: SummaryRequest):
    try:
        summary = compute_summary(body.dataset_id)
        return {
            "meta": {"dataset_id": body.dataset_id},
            "result": {"statistics": summary},
            "report": {},
            "artifacts": {}
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/groupby")
def eda_groupby(body: GroupbyRequest):
    try:
        result = compute_groupby(body.dataset_id, body.by, body.metrics)
        return {
            "meta": {"dataset_id": body.dataset_id, "grouped_by": body.by},
            "result": {"aggregations": result},
            "report": {},
            "artifacts": {}
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/correlation")
def eda_correlation(body: CorrelationRequest):
    try:
        corr = compute_correlation(body.dataset_id)
        return {
            "meta": {"dataset_id": body.dataset_id},
            "result": corr,
            "report": {},
            "artifacts": {}
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/plots")
def eda_plots(body: PlotsRequest):
    try:
        plots = generate_plots(body.dataset_id)
        return {
            "meta": {"dataset_id": body.dataset_id},
            "result": {},
            "report": {},
            "artifacts": plots
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))