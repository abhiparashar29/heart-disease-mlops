"""
FastAPI service for heart disease risk prediction.

Endpoints:
    GET  /            service info
    GET  /health      liveness / readiness check (used by Kubernetes)
    GET  /model-info  metadata of the loaded model
    POST /predict     predict heart disease risk for one patient
    GET  /metrics     Prometheus metrics

Run locally:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from src.preprocessing import FEATURES, ROOT

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(ROOT / "models" / "model.joblib")))
METADATA_PATH = MODEL_PATH.with_name("metadata.json")
QUIET_PATHS = {"/metrics", "/health"}  # polled constantly -> logged at DEBUG only

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("heart-api")

# Custom Prometheus metrics (in addition to the default HTTP metrics)
PREDICTIONS = Counter(
    "heart_predictions_total", "Number of predictions by predicted class", ["prediction"]
)
DISEASE_PROBABILITY = Histogram(
    "heart_disease_probability",
    "Distribution of predicted probability of heart disease",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

state: dict = {"model": None, "metadata": {}}


def load_model() -> None:
    try:
        state["model"] = joblib.load(MODEL_PATH)
        if METADATA_PATH.exists():
            state["metadata"] = json.loads(METADATA_PATH.read_text())
        logger.info(
            "Model loaded from %s (%s)",
            MODEL_PATH,
            state["metadata"].get("model_name", "unknown"),
        )
    except Exception:
        logger.exception("Could not load model from %s", MODEL_PATH)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Heart Disease Risk API",
    description="Predicts the risk of heart disease from patient health data "
    "(UCI Heart Disease dataset).",
    version="1.0.0",
    lifespan=lifespan,
)
Instrumentator().instrument(app).expose(app, include_in_schema=False)


class PatientData(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=50, le=250, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=50, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: float = Field(..., ge=50, le=250, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=1, le=3, description="Slope of peak exercise ST segment")
    ca: Optional[int] = Field(None, ge=0, le=3, description="Major vessels (0-3); may be missing")
    thal: Optional[Literal[3, 6, 7]] = Field(
        None, description="3 normal, 6 fixed defect, 7 reversible defect; may be missing"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
                "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
                "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    confidence: float
    probability_disease: float
    risk_level: str
    model_name: str

    model_config = {"protected_namespaces": ()}


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = uuid.uuid4().hex[:8]
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id

    level = logging.DEBUG if request.url.path in QUIET_PATHS else logging.INFO
    logger.log(level, json.dumps({
        "event": "request",
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "duration_ms": round(duration_ms, 2),
        "client": request.client.host if request.client else None,
    }))
    return response


def risk_level(probability: float) -> str:
    if probability < 0.3:
        return "low"
    if probability < 0.7:
        return "moderate"
    return "high"


@app.get("/")
def root():
    return {
        "service": "Heart Disease Risk API",
        "version": app.version,
        "docs": "/docs",
        "endpoints": ["/health", "/model-info", "/predict", "/metrics"],
    }


@app.get("/health")
def health():
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "ok",
        "model_loaded": True,
        "model_name": state["metadata"].get("model_name"),
    }


@app.get("/model-info")
def model_info():
    if not state["metadata"]:
        raise HTTPException(status_code=404, detail="Model metadata not available")
    return state["metadata"]


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientData):
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = pd.DataFrame([patient.model_dump()])[FEATURES].astype(float)
    p_disease = float(model.predict_proba(X)[0, 1])
    prediction = int(p_disease >= 0.5)

    result = PredictionResponse(
        prediction=prediction,
        label="Disease" if prediction else "No disease",
        confidence=round(max(p_disease, 1 - p_disease), 4),
        probability_disease=round(p_disease, 4),
        risk_level=risk_level(p_disease),
        model_name=state["metadata"].get("model_name", "unknown"),
    )

    PREDICTIONS.labels(prediction=str(prediction)).inc()
    DISEASE_PROBABILITY.observe(p_disease)
    logger.info(json.dumps({
        "event": "prediction",
        "input": patient.model_dump(),
        "prediction": prediction,
        "probability_disease": result.probability_disease,
        "risk_level": result.risk_level,
    }))
    return result
