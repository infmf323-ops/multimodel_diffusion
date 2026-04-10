import base64
import json
import logging
import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from .database import (
    create_generation_job,
    fetch_generation_by_id,
    fetch_generation_job,
    fetch_generations_by_batch,
    fetch_recent_generations,
    fetch_recent_generation_jobs,
    fetch_recent_predictions,
    fetch_synthetic_dataset_runs,
    fetch_synthetic_dataset_samples,
    healthcheck_db,
    init_db,
    insert_generation,
    insert_prediction,
)
from .diffusion_runtime import DiffusionLoraGenerator
from .model_runtime import CaptionRouterModel


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("inference-service")

REQUEST_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method", "path", "status"])
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency", ["method", "path"])
PREDICTION_COUNT = Counter("ml_predictions_total", "Predictions by label", ["label"])
PREDICTION_LATENCY = Histogram("ml_prediction_latency_seconds", "Prediction latency")
GENERATION_COUNT = Counter("ml_generations_total", "Diffusion generations by status", ["status"])
GENERATION_LATENCY = Histogram("ml_generation_latency_seconds", "Diffusion generation latency")
DB_HEALTH = Gauge("db_health_status", "Database health status")
MODEL_LOADED = Gauge("model_loaded_status", "Model loaded status")
DIFFUSION_MODEL_LOADED = Gauge("diffusion_model_loaded_status", "Diffusion LoRA model loaded status")

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.getenv("MODEL_PATH", REPO_ROOT / "lab3" / "artifacts" / "models" / "final_caption_router_model.npz"))
MODEL_META_PATH = Path(os.getenv("MODEL_META_PATH", REPO_ROOT / "lab3" / "artifacts" / "models" / "final_caption_router_model_meta.json"))
SAMPLE_DATA_PATH = Path(os.getenv("SAMPLE_DATA_PATH", REPO_ROOT / "lab3" / "data" / "conceptual_captions_sample_100.tsv"))
SUMMARY_PATH = Path(os.getenv("SUMMARY_PATH", REPO_ROOT / "lab3" / "artifacts" / "lab3_summary.json"))
EXPERIMENTS_PATH = Path(os.getenv("EXPERIMENTS_PATH", REPO_ROOT / "lab3" / "artifacts" / "experiments" / "experiment_log.jsonl"))
DIFFUSION_BASE_MODEL = os.getenv("DIFFUSION_BASE_MODEL", "segmind/tiny-sd")
LORA_ADAPTER_PATH = Path(
    os.getenv(
        "LORA_ADAPTER_PATH",
        REPO_ROOT / "lora_experiment" / "artifacts" / "lora_ab_cc_seed_60steps_clip" / "runs" / "seed_plus_synthetic" / "adapters",
    )
)
GENERATION_OUTPUT_DIR = Path(os.getenv("GENERATION_OUTPUT_DIR", REPO_ROOT / "generated_outputs"))
DIFFUSION_DEVICE = os.getenv("DIFFUSION_DEVICE", "auto")


class PredictRequest(BaseModel):
    caption: str = Field(..., min_length=3, max_length=500)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=500)
    negative_prompt: str = Field(
        default="low quality, blurry, distorted, watermark, text artifacts",
        max_length=500,
    )
    seed: int | None = Field(default=None, ge=0, le=2_147_483_647)
    width: int = Field(default=256, ge=128, le=512)
    height: int = Field(default=256, ge=128, le=512)
    num_inference_steps: int = Field(default=12, ge=1, le=50)
    guidance_scale: float = Field(default=5.0, ge=1.0, le=20.0)


class BatchGenerateRequest(BaseModel):
    topic: str = Field(..., min_length=2, max_length=120)
    count: int = Field(default=20, ge=1, le=50)
    preview_limit: int = Field(default=5, ge=0, le=20)
    negative_prompt: str = Field(
        default="low quality, blurry, distorted, watermark, text artifacts",
        max_length=500,
    )
    seed: int | None = Field(default=None, ge=0, le=2_147_483_000)
    width: int = Field(default=256, ge=128, le=512)
    height: int = Field(default=256, ge=128, le=512)
    num_inference_steps: int = Field(default=12, ge=1, le=50)
    guidance_scale: float = Field(default=5.0, ge=1.0, le=20.0)


class GenerationJobRequest(BaseModel):
    topic: str = Field(..., min_length=2, max_length=120)
    count: int = Field(default=100, ge=1, le=1000)
    preview_limit: int = Field(default=5, ge=0, le=20)
    negative_prompt: str = Field(
        default="low quality, blurry, distorted, watermark, text artifacts",
        max_length=500,
    )
    seed: int | None = Field(default=None, ge=0, le=2_147_483_000)
    width: int = Field(default=256, ge=128, le=512)
    height: int = Field(default=256, ge=128, le=512)
    num_inference_steps: int = Field(default=12, ge=1, le=50)
    guidance_scale: float = Field(default=5.0, ge=1.0, le=20.0)


app = FastAPI(
    title="Multimodal Diffusion LoRA Service",
    version="2.0.0",
    description="Inference API for the LR3 LoRA diffusion adapter with monitoring and PostgreSQL history.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = CaptionRouterModel(MODEL_PATH, MODEL_META_PATH, SAMPLE_DATA_PATH)
MODEL_LOADED.set(1)
diffusion_generator = DiffusionLoraGenerator(
    base_model_id=DIFFUSION_BASE_MODEL,
    lora_adapter_path=LORA_ADAPTER_PATH,
    output_dir=GENERATION_OUTPUT_DIR,
    device=DIFFUSION_DEVICE,
)
DIFFUSION_MODEL_LOADED.set(0)

TOPIC_TRANSLATIONS = {
    "мебель": "furniture",
    "стулья": "chairs",
    "стул": "chair",
    "столы": "tables",
    "стол": "table",
    "диваны": "sofas",
    "диван": "sofa",
    "шкаф": "wardrobe",
    "шкафы": "wardrobes",
    "лампы": "lamps",
    "лампа": "lamp",
    "посуда": "tableware",
    "обувь": "shoes",
    "одежда": "clothing",
    "растения": "plants",
    "растение": "plant",
    "животные": "animals",
    "машины": "cars",
    "машина": "car",
}

TOPIC_PROMPT_TEMPLATES = [
    "studio product photo of {topic}, clean neutral background, soft natural lighting, high detail",
    "modern catalog image of {topic}, centered composition, realistic materials, warm daylight",
    "minimal interior scene focused on {topic}, balanced composition, realistic shadows",
    "close-up commercial photo of {topic}, crisp details, shallow depth of field, no people",
    "lifestyle photo of {topic} in a cozy apartment, natural colors, realistic scene",
    "front view of {topic}, simple background, high quality product photography",
    "angled view of {topic}, textured surfaces, soft studio lighting, realistic proportions",
    "set of varied {topic}, organized composition, neutral background, high detail",
]


def build_topic_prompts(topic: str, count: int) -> list[str]:
    clean_topic = " ".join(topic.strip().split())
    prompt_topic = TOPIC_TRANSLATIONS.get(clean_topic.lower(), clean_topic)
    return [
        TOPIC_PROMPT_TEMPLATES[index % len(TOPIC_PROMPT_TEMPLATES)].format(topic=prompt_topic)
        for index in range(count)
    ]


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
        "generate": "/generate",
        "generate_batch": "/generate/batch",
        "generation_jobs": "/generation-jobs",
    }


@app.get("/health")
def health():
    db_ok = healthcheck_db()
    DB_HEALTH.set(1 if db_ok else 0)
    return {
        "status": "ok" if db_ok else "degraded",
        "model_loaded": True,
        "diffusion_model_loaded": diffusion_generator.is_loaded(),
        "diffusion_model": diffusion_generator.info(),
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


@app.post("/generate")
def generate(payload: GenerateRequest):
    prompt = payload.prompt.strip()
    negative_prompt = payload.negative_prompt.strip()
    started = time.perf_counter()

    try:
        result = diffusion_generator.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=payload.seed,
            width=payload.width,
            height=payload.height,
            num_inference_steps=payload.num_inference_steps,
            guidance_scale=payload.guidance_scale,
        )
    except Exception as exc:
        GENERATION_COUNT.labels("error").inc()
        logger.error(json.dumps({"event": "generation_failed", "error": str(exc)}, ensure_ascii=False))
        raise HTTPException(status_code=500, detail=f"Diffusion generation failed: {exc}")

    latency_ms = (time.perf_counter() - started) * 1000.0
    GENERATION_COUNT.labels("success").inc()
    GENERATION_LATENCY.observe(latency_ms / 1000.0)
    DIFFUSION_MODEL_LOADED.set(1 if diffusion_generator.is_loaded() else 0)

    try:
        generation_id = insert_generation(prompt, negative_prompt, result, latency_ms)
    except Exception as exc:
        logger.error(json.dumps({"event": "generation_db_insert_failed", "error": str(exc)}, ensure_ascii=False))
        raise HTTPException(status_code=500, detail="Generation was computed, but storing history failed.")

    image_base64 = base64.b64encode(result.image_bytes).decode("ascii")
    return {
        "generation_id": generation_id,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image_base64": image_base64,
        "mime_type": "image/png",
        "seed": result.seed,
        "base_model_checkpoint": result.model_id,
        "lora_adapter_path": result.lora_adapter_path,
        "output_path": result.output_path,
        "device": result.device,
        "width": result.width,
        "height": result.height,
        "num_inference_steps": result.num_inference_steps,
        "guidance_scale": result.guidance_scale,
        "latency_ms": round(latency_ms, 4),
    }


@app.post("/generate/batch")
def generate_batch(payload: BatchGenerateRequest):
    topic = payload.topic.strip()
    negative_prompt = payload.negative_prompt.strip()
    prompts = build_topic_prompts(topic, payload.count)
    base_seed = payload.seed if payload.seed is not None else int(time.time() * 1000) % 2_147_483_000
    batch_id = f"webui-batch-{uuid.uuid4().hex[:12]}"
    started = time.perf_counter()
    items = []

    for index, prompt in enumerate(prompts):
        item_seed = base_seed + index
        item_started = time.perf_counter()
        try:
            result = diffusion_generator.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=item_seed,
                width=payload.width,
                height=payload.height,
                num_inference_steps=payload.num_inference_steps,
                guidance_scale=payload.guidance_scale,
            )
        except Exception as exc:
            GENERATION_COUNT.labels("error").inc()
            logger.error(
                json.dumps(
                    {
                        "event": "batch_generation_failed",
                        "topic": topic,
                        "prompt": prompt,
                        "index": index,
                        "error": str(exc),
                    },
                    ensure_ascii=False,
                )
            )
            raise HTTPException(status_code=500, detail=f"Batch diffusion generation failed at item {index + 1}: {exc}")

        item_latency_ms = (time.perf_counter() - item_started) * 1000.0
        GENERATION_COUNT.labels("success").inc()
        GENERATION_LATENCY.observe(item_latency_ms / 1000.0)
        DIFFUSION_MODEL_LOADED.set(1 if diffusion_generator.is_loaded() else 0)

        try:
            generation_id = insert_generation(
                prompt,
                negative_prompt,
                result,
                item_latency_ms,
                batch_id=batch_id,
                topic=topic,
                source="batch_topic",
            )
        except Exception as exc:
            logger.error(json.dumps({"event": "batch_generation_db_insert_failed", "error": str(exc)}, ensure_ascii=False))
            raise HTTPException(status_code=500, detail="Batch generation was computed, but storing history failed.")

        items.append(
            {
                "generation_id": generation_id,
                "batch_id": batch_id,
                "topic": topic,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image_base64": base64.b64encode(result.image_bytes).decode("ascii") if index < payload.preview_limit else None,
                "mime_type": "image/png",
                "has_preview": index < payload.preview_limit,
                "seed": result.seed,
                "base_model_checkpoint": result.model_id,
                "lora_adapter_path": result.lora_adapter_path,
                "output_path": result.output_path,
                "device": result.device,
                "width": result.width,
                "height": result.height,
                "num_inference_steps": result.num_inference_steps,
                "guidance_scale": result.guidance_scale,
            }
        )

    total_latency_ms = (time.perf_counter() - started) * 1000.0
    return {
        "topic": topic,
        "count_requested": payload.count,
        "count_generated": len(items),
        "batch_id": batch_id,
        "base_seed": base_seed,
        "preview_limit": payload.preview_limit,
        "latency_ms": round(total_latency_ms, 4),
        "prompt_strategy": "topic_templates_for_dataset_gap_augmentation",
        "items": items,
    }


def generation_job_response(job):
    if not job:
        raise HTTPException(status_code=404, detail="Generation job not found.")
    items = fetch_generations_by_batch(job["job_id"], limit=max(job["count_requested"], job["preview_limit"], 1))
    preview_items = []
    for item in items[: job["preview_limit"]]:
        preview_item = dict(item)
        preview_item["preview_url"] = f"/generations/{item['id']}/image"
        preview_items.append(preview_item)
    return {
        **job,
        "progress": {
            "generated": job["count_generated"],
            "requested": job["count_requested"],
            "percent": round((job["count_generated"] / job["count_requested"]) * 100, 2)
            if job["count_requested"]
            else 0,
        },
        "items": items,
        "preview_items": preview_items,
        "manifest_url": f"/generation-jobs/{job['job_id']}/manifest" if job["manifest_path"] else None,
    }


@app.post("/generation-jobs")
def create_job(payload: GenerationJobRequest):
    topic = payload.topic.strip()
    base_seed = payload.seed if payload.seed is not None else int(time.time() * 1000) % 2_147_483_000
    job_id = f"job-{uuid.uuid4().hex[:12]}"
    job = create_generation_job(
        job_id=job_id,
        topic=topic,
        count_requested=payload.count,
        preview_limit=payload.preview_limit,
        negative_prompt=payload.negative_prompt.strip(),
        base_seed=base_seed,
        width=payload.width,
        height=payload.height,
        num_inference_steps=payload.num_inference_steps,
        guidance_scale=payload.guidance_scale,
        base_model_checkpoint=DIFFUSION_BASE_MODEL,
        lora_adapter_path=str(LORA_ADAPTER_PATH),
        metadata={
            "prompt_strategy": "topic_templates_for_dataset_gap_augmentation",
            "queue_backend": "postgres_generation_jobs",
            "created_by": "webui",
        },
    )
    return generation_job_response(job)


@app.get("/generation-jobs")
def generation_jobs(limit: int = Query(default=20, ge=1, le=100)):
    return {"items": [generation_job_response(job) for job in fetch_recent_generation_jobs(limit)]}


@app.get("/generation-jobs/{job_id}")
def generation_job_status(job_id: str):
    return generation_job_response(fetch_generation_job(job_id))


@app.get("/generation-jobs/{job_id}/manifest")
def generation_job_manifest(job_id: str):
    job = fetch_generation_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Generation job not found.")
    if not job["manifest_path"]:
        raise HTTPException(status_code=404, detail="Manifest is not ready yet.")
    manifest_path = Path(job["manifest_path"])
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Manifest file is missing from generated outputs volume.")
    return FileResponse(
        manifest_path,
        media_type="text/csv",
        filename=f"{job_id}_manifest.csv",
    )


@app.get("/generations/{generation_id}/image")
def generation_image(generation_id: int):
    generation = fetch_generation_by_id(generation_id)
    if not generation:
        raise HTTPException(status_code=404, detail="Generation not found.")
    output_path = Path(generation["output_path"])
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Generated image file is missing from output volume.")
    return FileResponse(output_path, media_type="image/png", filename=output_path.name)


@app.get("/history")
def history(limit: int = Query(default=20, ge=1, le=100)):
    return {"items": fetch_recent_predictions(limit)}


@app.get("/generations/history")
def generations_history(limit: int = Query(default=20, ge=1, le=100)):
    return {"items": fetch_recent_generations(limit)}


@app.get("/synthetic-dataset/runs")
def synthetic_dataset_runs(limit: int = Query(default=20, ge=1, le=100)):
    return {"items": fetch_synthetic_dataset_runs(limit)}


@app.get("/synthetic-dataset/samples")
def synthetic_dataset_samples(limit: int = Query(default=20, ge=1, le=100)):
    return {"items": fetch_synthetic_dataset_samples(limit)}


@app.get("/model/info")
def model_info():
    return {
        "caption_router": model.info(),
        "diffusion_lora": diffusion_generator.info(),
    }


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
