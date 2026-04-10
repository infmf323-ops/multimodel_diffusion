import csv
import json
import logging
import os
import socket
import time
from pathlib import Path

from .database import (
    acquire_next_generation_job,
    complete_generation_job,
    fail_generation_job,
    fetch_generations_by_batch,
    init_db,
    insert_generation,
    update_generation_job_progress,
)
from .diffusion_runtime import DiffusionLoraGenerator


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("generation-worker")

REPO_ROOT = Path(__file__).resolve().parents[2]
DIFFUSION_BASE_MODEL = os.getenv("DIFFUSION_BASE_MODEL", "segmind/tiny-sd")
LORA_ADAPTER_PATH = Path(
    os.getenv(
        "LORA_ADAPTER_PATH",
        REPO_ROOT / "lora_experiment" / "artifacts" / "lora_ab_cc_seed_60steps_clip" / "runs" / "seed_plus_synthetic" / "adapters",
    )
)
GENERATION_OUTPUT_DIR = Path(os.getenv("GENERATION_OUTPUT_DIR", REPO_ROOT / "generated_outputs"))
DIFFUSION_DEVICE = os.getenv("DIFFUSION_DEVICE", "auto")
POLL_SECONDS = float(os.getenv("GENERATION_WORKER_POLL_SECONDS", "2"))

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


def write_manifest(job) -> str:
    manifest_dir = GENERATION_OUTPUT_DIR / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{job['job_id']}_manifest.csv"
    rows = fetch_generations_by_batch(job["job_id"], limit=job["count_requested"])
    fieldnames = [
        "generation_id",
        "batch_id",
        "topic",
        "prompt",
        "negative_prompt",
        "seed",
        "base_model_checkpoint",
        "lora_adapter_path",
        "output_path",
        "width",
        "height",
        "num_inference_steps",
        "guidance_scale",
        "device",
        "latency_ms",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "generation_id": row["id"],
                    "batch_id": row["batch_id"],
                    "topic": row["topic"],
                    "prompt": row["prompt"],
                    "negative_prompt": row["negative_prompt"],
                    "seed": row["seed"],
                    "base_model_checkpoint": row["base_model_checkpoint"],
                    "lora_adapter_path": row["lora_adapter_path"],
                    "output_path": row["output_path"],
                    "width": row["width"],
                    "height": row["height"],
                    "num_inference_steps": row["num_inference_steps"],
                    "guidance_scale": row["guidance_scale"],
                    "device": row["device"],
                    "latency_ms": row["latency_ms"],
                }
            )
    return str(manifest_path)


def process_job(generator: DiffusionLoraGenerator, job):
    prompts = build_topic_prompts(job["topic"], job["count_requested"])
    logger.info(json.dumps({"event": "job_started", "job_id": job["job_id"], "count": len(prompts)}, ensure_ascii=False))

    for index, prompt in enumerate(prompts):
        item_started = time.perf_counter()
        result = generator.generate(
            prompt=prompt,
            negative_prompt=job["negative_prompt"],
            seed=job["base_seed"] + index,
            width=job["width"],
            height=job["height"],
            num_inference_steps=job["num_inference_steps"],
            guidance_scale=job["guidance_scale"],
        )
        latency_ms = (time.perf_counter() - item_started) * 1000.0
        generation_id = insert_generation(
            prompt,
            job["negative_prompt"],
            result,
            latency_ms,
            batch_id=job["job_id"],
            topic=job["topic"],
            source="async_job",
        )
        update_generation_job_progress(job["job_id"], index + 1)
        logger.info(
            json.dumps(
                {
                    "event": "job_item_generated",
                    "job_id": job["job_id"],
                    "generation_id": generation_id,
                    "progress": f"{index + 1}/{len(prompts)}",
                },
                ensure_ascii=False,
            )
        )

    manifest_path = write_manifest(job)
    complete_generation_job(job["job_id"], manifest_path)
    logger.info(json.dumps({"event": "job_done", "job_id": job["job_id"], "manifest_path": manifest_path}, ensure_ascii=False))


def main():
    worker_id = os.getenv("GENERATION_WORKER_ID", f"{socket.gethostname()}-{os.getpid()}")
    init_db()
    generator = DiffusionLoraGenerator(
        base_model_id=DIFFUSION_BASE_MODEL,
        lora_adapter_path=LORA_ADAPTER_PATH,
        output_dir=GENERATION_OUTPUT_DIR,
        device=DIFFUSION_DEVICE,
    )
    logger.info(json.dumps({"event": "worker_started", "worker_id": worker_id}, ensure_ascii=False))

    while True:
        job = acquire_next_generation_job(worker_id)
        if not job:
            time.sleep(POLL_SECONDS)
            continue
        try:
            process_job(generator, job)
        except Exception as exc:
            fail_generation_job(job["job_id"], str(exc))
            logger.exception(json.dumps({"event": "job_failed", "job_id": job["job_id"], "error": str(exc)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
