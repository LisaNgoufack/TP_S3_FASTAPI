from fastapi import FastAPI
from app.routers import clean, eda, mv, ml, ml_avance

app = FastAPI(title="TP1-TP2-TP3-TP4-TP5 API")

app.include_router(clean.router)
app.include_router(eda.router)
app.include_router(mv.router)
app.include_router(ml.router)
app.include_router(ml_avance.router)

@app.get("/")
def read_root():
    return {"message": "Pipeline ML Complet - 5 TPs terminés !"}