import base64
import json
import logging
import os
import statistics
import time
import uuid
from itertools import combinations
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from PIL import Image, ImageFilter, ImageStat
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
from .diffusion_runtime import DiffusionGenerator
from .model_runtime import CaptionRouterModel
from .storage import ObjectStorage


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
DIFFUSION_MODEL_LOADED = Gauge("diffusion_model_loaded_status", "Diffusion model loaded status")

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.getenv("MODEL_PATH", REPO_ROOT / "lab3" / "artifacts" / "models" / "final_caption_router_model.npz"))
MODEL_META_PATH = Path(os.getenv("MODEL_META_PATH", REPO_ROOT / "lab3" / "artifacts" / "models" / "final_caption_router_model_meta.json"))
SAMPLE_DATA_PATH = Path(os.getenv("SAMPLE_DATA_PATH", REPO_ROOT / "lab3" / "data" / "conceptual_captions_sample_100.tsv"))
SUMMARY_PATH = Path(os.getenv("SUMMARY_PATH", REPO_ROOT / "synthetic_dataset" / "parameter_experiments" / "summary.json"))
EXPERIMENTS_PATH = Path(os.getenv("EXPERIMENTS_PATH", REPO_ROOT / "lab3" / "artifacts" / "experiments" / "experiment_log.jsonl"))
DIFFUSION_BASE_MODEL = os.getenv("DIFFUSION_BASE_MODEL", "segmind/tiny-sd")
ADAPTER_PATH_ENV = os.getenv("DIFFUSION_ADAPTER_PATH") or os.getenv("LORA_ADAPTER_PATH")
DIFFUSION_ADAPTER_PATH = Path(ADAPTER_PATH_ENV) if ADAPTER_PATH_ENV else None
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
    width: int = Field(default=320, ge=128, le=512)
    height: int = Field(default=320, ge=128, le=512)
    num_inference_steps: int = Field(default=22, ge=1, le=50)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)


class GenerationJobRequest(BaseModel):
    topic: str = Field(..., min_length=2, max_length=120)
    count: int = Field(default=100, ge=1, le=1000)
    preview_limit: int = Field(default=5, ge=0, le=20)
    # Сколько строк вернуть в поле items ответа (0 = только превью-картинки, без таблицы метаданных)
    response_items_limit: int = Field(default=32, ge=0, le=500)
    negative_prompt: str = Field(
        default="low quality, blurry, distorted, watermark, text artifacts",
        max_length=500,
    )
    seed: int | None = Field(default=None, ge=0, le=2_147_483_000)
    width: int = Field(default=320, ge=128, le=512)
    height: int = Field(default=320, ge=128, le=512)
    num_inference_steps: int = Field(default=22, ge=1, le=50)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)


app = FastAPI(
    title="Multimodal Diffusion Dataset Service",
    version="2.0.0",
    description="Inference API for synthetic dataset generation with monitoring and PostgreSQL history.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = CaptionRouterModel(MODEL_PATH, MODEL_META_PATH, SAMPLE_DATA_PATH)
MODEL_LOADED.set(1)
diffusion_generator = DiffusionGenerator(
    base_model_id=DIFFUSION_BASE_MODEL,
    adapter_path=DIFFUSION_ADAPTER_PATH,
    output_dir=GENERATION_OUTPUT_DIR,
    device=DIFFUSION_DEVICE,
)
object_storage = ObjectStorage()
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


def upload_generation_artifact(result, batch_id: str | None = None) -> str | None:
    file_name = Path(result.output_path).name
    object_key = f"{object_storage.prefix}/images/{file_name}"
    if batch_id:
        object_key = f"{object_storage.prefix}/batches/{batch_id}/images/{file_name}"
    return object_storage.upload_bytes(result.image_bytes, object_key, "image/png")


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
        output_object_uri = upload_generation_artifact(result)
        generation_id = insert_generation(
            prompt,
            negative_prompt,
            result,
            latency_ms,
            output_object_uri=output_object_uri,
        )
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
        "adapter_path": result.lora_adapter_path or None,
        "lora_adapter_path": result.lora_adapter_path,
        "output_path": result.output_path,
        "output_object_uri": output_object_uri,
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
            output_object_uri = upload_generation_artifact(result, batch_id=batch_id)
            generation_id = insert_generation(
                prompt,
                negative_prompt,
                result,
                item_latency_ms,
                batch_id=batch_id,
                topic=topic,
                source="batch_topic",
                output_object_uri=output_object_uri,
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
                "adapter_path": result.lora_adapter_path or None,
                "lora_adapter_path": result.lora_adapter_path,
                "output_path": result.output_path,
                "output_object_uri": output_object_uri,
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


def _preview_items_from_rows(rows: list[dict], preview_limit: int) -> list[dict]:
    preview_items = []
    for item in rows[:preview_limit]:
        preview_item = dict(item)
        preview_item["preview_url"] = f"/generations/{item['id']}/image"
        preview_items.append(preview_item)
    return preview_items


def _average_hash(image: Image.Image, hash_size: int = 8) -> list[int]:
    gray = image.convert("L").resize((hash_size, hash_size))
    pixels = list(gray.getdata())
    avg = sum(pixels) / len(pixels)
    return [1 if pixel >= avg else 0 for pixel in pixels]


def _technical_quality_score(image_path: Path) -> float | None:
    try:
        with Image.open(image_path) as image:
            gray = image.convert("L")
            hsv = image.convert("HSV")
            gray_stat = ImageStat.Stat(gray)
            hsv_stat = ImageStat.Stat(hsv)
            edge_stat = ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES))

            contrast = min(float(gray_stat.stddev[0]) / 64.0, 1.0)
            saturation = min(float(hsv_stat.mean[1]) / 255.0, 1.0)
            edge_density = min(float(edge_stat.mean[0]) / 42.0, 1.0)
            brightness = float(gray_stat.mean[0]) / 255.0
            brightness_penalty = abs(brightness - 0.55) / 0.55

            score = 0.4 * contrast + 0.35 * saturation + 0.25 * edge_density
            score = max(0.0, min(1.0, score * (1.0 - 0.3 * brightness_penalty)))
            return round(score, 4)
    except Exception:
        return None


def _dataset_metrics_from_rows(job: dict, rows: list[dict]) -> dict:
    total_rows = len(rows)
    metrics = {
        "sample_count": total_rows,
        "metrics_scope": "full_batch" if job["count_generated"] <= 1000 else "sampled",
        "avg_latency_ms": None,
        "min_latency_ms": None,
        "max_latency_ms": None,
        "images_per_minute": None,
        "avg_file_size_kb": None,
        "avg_quality_score": None,
        "high_quality_share": None,
        "prompt_uniqueness_ratio": None,
        "near_duplicate_rate": None,
    }
    if total_rows == 0:
        return metrics

    latencies = [float(item["latency_ms"]) for item in rows]
    metrics["avg_latency_ms"] = round(statistics.mean(latencies), 2)
    metrics["min_latency_ms"] = round(min(latencies), 2)
    metrics["max_latency_ms"] = round(max(latencies), 2)
    total_generation_time_s = sum(latencies) / 1000.0
    if total_generation_time_s > 0:
        metrics["images_per_minute"] = round((total_rows / total_generation_time_s) * 60.0, 2)
    metrics["prompt_uniqueness_ratio"] = round(len({item["prompt"] for item in rows}) / total_rows, 4)

    file_sizes_kb = []
    quality_scores = []
    hashes = []
    for row in rows:
        output_path = Path(row["output_path"])
        if not output_path.exists():
            continue
        file_sizes_kb.append(round(output_path.stat().st_size / 1024.0, 3))
        quality_score = _technical_quality_score(output_path)
        if quality_score is not None:
            quality_scores.append(quality_score)
        try:
            with Image.open(output_path) as image:
                hashes.append(_average_hash(image))
        except Exception:
            pass

    if file_sizes_kb:
        metrics["avg_file_size_kb"] = round(statistics.mean(file_sizes_kb), 2)
    if quality_scores:
        metrics["avg_quality_score"] = round(statistics.mean(quality_scores), 4)
        metrics["high_quality_share"] = round(
            sum(1 for score in quality_scores if score >= 0.6) / len(quality_scores), 4
        )

    duplicate_pairs = 0
    total_pairs = 0
    for left, right in combinations(hashes, 2):
        total_pairs += 1
        distance = sum(1 for a, b in zip(left, right) if a != b)
        if distance <= 5:
            duplicate_pairs += 1
    metrics["near_duplicate_rate"] = round(duplicate_pairs / total_pairs, 4) if total_pairs else 0.0
    return metrics


def _items_summary(job: dict, items: list[dict]) -> dict:
    total = job["count_generated"]
    n = len(items)
    summary = {
        "items_total_in_batch": total,
        "items_returned": n,
        "items_truncated": total > n,
        "manifest_complete": bool(job.get("manifest_path")),
    }
    if items:
        latencies = [float(i["latency_ms"]) for i in items]
        summary["sample_avg_latency_ms"] = round(sum(latencies) / len(latencies), 4)
        summary["first_generation_id"] = items[0]["id"]
        summary["last_generation_id"] = items[-1]["id"]
    return summary


def generation_job_response(job, items_limit: int = 48):
    """items_limit: max DB rows returned for UI (capped). Use 0 to skip metadata rows (previews only)."""
    if not job:
        raise HTTPException(status_code=404, detail="Generation job not found.")
    pl = max(int(job["preview_limit"] or 0), 0)
    metrics_rows = fetch_generations_by_batch(job["job_id"], limit=1000)
    dataset_metrics = _dataset_metrics_from_rows(job, metrics_rows)
    if items_limit <= 0:
        fetch_limit = max(pl, 1)
        rows = fetch_generations_by_batch(job["job_id"], limit=min(fetch_limit, 500))
        preview_items = _preview_items_from_rows(rows, pl)
        table_items: list[dict] = []
    else:
        fetch_limit = min(max(pl, items_limit), 500)
        rows = fetch_generations_by_batch(job["job_id"], limit=fetch_limit)
        preview_items = _preview_items_from_rows(rows, pl)
        table_items = rows
    return {
        **job,
        "progress": {
            "generated": job["count_generated"],
            "requested": job["count_requested"],
            "percent": round((job["count_generated"] / job["count_requested"]) * 100, 2)
            if job["count_requested"]
            else 0,
        },
        "items": table_items,
        "preview_items": preview_items,
        "dataset_metrics": dataset_metrics,
        "items_summary": {
            **(_items_summary(job, rows)),
            "metadata_table_rows": len(table_items),
        },
        "manifest_url": f"/generation-jobs/{job['job_id']}/manifest" if job["manifest_path"] else None,
        "manifest_object_uri": job["manifest_object_uri"],
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
        lora_adapter_path=str(DIFFUSION_ADAPTER_PATH) if DIFFUSION_ADAPTER_PATH else "",
        metadata={
            "prompt_strategy": "topic_templates_for_dataset_gap_augmentation",
            "queue_backend": "postgres_generation_jobs",
            "created_by": "webui",
        },
    )
    return generation_job_response(job, items_limit=payload.response_items_limit)


@app.get("/generation-jobs")
def generation_jobs(
    limit: int = Query(default=20, ge=1, le=100),
    items_limit: int = Query(default=12, ge=0, le=500),
):
    return {
        "items": [
            generation_job_response(job, items_limit=items_limit)
            for job in fetch_recent_generation_jobs(limit)
        ]
    }


@app.get("/generation-jobs/{job_id}")
def generation_job_status(
    job_id: str,
    items_limit: int = Query(default=48, ge=0, le=500),
):
    return generation_job_response(fetch_generation_job(job_id), items_limit=items_limit)


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
        "diffusion_model": diffusion_generator.info(),
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
