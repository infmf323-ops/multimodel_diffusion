import json
import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from .database import fetch_recent_predictions, healthcheck_db, init_db, insert_prediction
from .model_runtime import CaptionRouterModel


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("inference-service")

REQUEST_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method", "path", "status"])
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency", ["method", "path"])
PREDICTION_COUNT = Counter("ml_predictions_total", "Predictions by label", ["label"])
PREDICTION_LATENCY = Histogram("ml_prediction_latency_seconds", "Prediction latency")
DB_HEALTH = Gauge("db_health_status", "Database health status")
MODEL_LOADED = Gauge("model_loaded_status", "Model loaded status")

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.getenv("MODEL_PATH", REPO_ROOT / "lab3" / "artifacts" / "models" / "final_caption_router_model.npz"))
MODEL_META_PATH = Path(os.getenv("MODEL_META_PATH", REPO_ROOT / "lab3" / "artifacts" / "models" / "final_caption_router_model_meta.json"))
SAMPLE_DATA_PATH = Path(os.getenv("SAMPLE_DATA_PATH", REPO_ROOT / "lab3" / "data" / "conceptual_captions_sample_100.tsv"))
SUMMARY_PATH = Path(os.getenv("SUMMARY_PATH", REPO_ROOT / "lab3" / "artifacts" / "lab3_summary.json"))
EXPERIMENTS_PATH = Path(os.getenv("EXPERIMENTS_PATH", REPO_ROOT / "lab3" / "artifacts" / "experiments" / "experiment_log.jsonl"))


class PredictRequest(BaseModel):
    caption: str = Field(..., min_length=3, max_length=500)


app = FastAPI(
    title="Multimodal Diffusion Caption Router",
    version="1.0.0",
    description="Inference API for the LR3 final model with monitoring and PostgreSQL history.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = CaptionRouterModel(MODEL_PATH, MODEL_META_PATH, SAMPLE_DATA_PATH)
MODEL_LOADED.set(1)


@app.middleware("http")
async def log_and_measure(request: Request, call_next):
    started = time.perf_counter()
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        duration = time.perf_counter() - started
        status = str(response.status_code if response else 500)
        path = request.url.path
        REQUEST_COUNT.labels(request.method, path, status).inc()
        REQUEST_LATENCY.labels(request.method, path).observe(duration)
        logger.info(
            json.dumps(
                {
                    "event": "request",
                    "method": request.method,
                    "path": path,
                    "status": int(status),
                    "duration_ms": round(duration * 1000, 3),
                    "client": request.client.host if request.client else None,
                },
                ensure_ascii=False,
            )
        )


@app.on_event("startup")
def startup():
    init_db()


@app.get("/")
def root():
    return {
        "service": "multimodal-diffusion-inference",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


@app.get("/health")
def health():
    db_ok = healthcheck_db()
    DB_HEALTH.set(1 if db_ok else 0)
    return {
        "status": "ok" if db_ok else "degraded",
        "model_loaded": True,
        "database_connected": db_ok,
        "model_labels": model.info()["labels"],
    }


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(payload: PredictRequest):
    caption = payload.caption.strip()
    started = time.perf_counter()
    prediction = model.predict(caption)
    latency_ms = (time.perf_counter() - started) * 1000.0
    PREDICTION_COUNT.labels(prediction["predicted_label"]).inc()
    PREDICTION_LATENCY.observe(latency_ms / 1000.0)

    try:
        insert_prediction(caption, prediction, latency_ms)
    except Exception as exc:
        logger.error(json.dumps({"event": "db_insert_failed", "error": str(exc)}, ensure_ascii=False))
        raise HTTPException(status_code=500, detail="Prediction was computed, but storing history failed.")

    return {
        "caption": caption,
        "predicted_label": prediction["predicted_label"],
        "confidence": round(prediction["confidence"], 6),
        "probabilities": prediction["probabilities"],
        "model_sha": prediction["model_sha"],
        "latency_ms": round(latency_ms, 4),
    }


@app.get("/history")
def history(limit: int = Query(default=20, ge=1, le=100)):
    return {"items": fetch_recent_predictions(limit)}


@app.get("/model/info")
def model_info():
    return model.info()


@app.get("/experiments/summary")
def experiments_summary():
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


@app.get("/experiments/logs")
def experiments_logs(limit: int = Query(default=20, ge=1, le=100)):
    rows = []
    with EXPERIMENTS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return {"items": rows[:limit]}


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception):
    logger.error(json.dumps({"event": "unhandled_exception", "error": str(exc)}, ensure_ascii=False))
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
