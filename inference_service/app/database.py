import json
import os
from contextlib import contextmanager

import psycopg


def dsn() -> str:
    return os.getenv(
        "DATABASE_URL",
        "postgresql://multimodal:multimodal@postgres:5432/multimodal_diffusion",
    )


@contextmanager
def get_connection():
    conn = psycopg.connect(dsn())
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_requests (
                    id BIGSERIAL PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    caption TEXT NOT NULL,
                    predicted_label TEXT NOT NULL,
                    confidence DOUBLE PRECISION NOT NULL,
                    probabilities JSONB NOT NULL,
                    model_sha TEXT NOT NULL,
                    latency_ms DOUBLE PRECISION NOT NULL
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS synthetic_dataset_runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    source_type TEXT NOT NULL,
                    generator TEXT NOT NULL,
                    base_model_checkpoint TEXT NOT NULL,
                    resolution TEXT NOT NULL,
                    num_inference_steps INTEGER NOT NULL,
                    guidance_scale DOUBLE PRECISION NOT NULL,
                    base_seed BIGINT NOT NULL,
                    device TEXT NOT NULL,
                    manifest_object_uri TEXT NOT NULL,
                    stats_object_uri TEXT NOT NULL,
                    preview_object_uri TEXT NOT NULL,
                    model_reference_object_uri TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS synthetic_dataset_samples (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES synthetic_dataset_runs(run_id) ON DELETE CASCADE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    prompt TEXT NOT NULL,
                    generation_prompt TEXT NOT NULL,
                    negative_prompt TEXT NOT NULL,
                    domain_tag TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    generator TEXT NOT NULL,
                    base_model_checkpoint TEXT NOT NULL,
                    seed BIGINT NOT NULL,
                    resolution TEXT NOT NULL,
                    num_inference_steps INTEGER NOT NULL,
                    guidance_scale DOUBLE PRECISION NOT NULL,
                    quality_score DOUBLE PRECISION NOT NULL,
                    quality_score_type TEXT NOT NULL,
                    quality_flag TEXT NOT NULL,
                    local_image_path TEXT NOT NULL,
                    image_object_uri TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS generation_requests (
                    id BIGSERIAL PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    batch_id TEXT,
                    source TEXT NOT NULL DEFAULT 'single_prompt',
                    topic TEXT,
                    prompt TEXT NOT NULL,
                    negative_prompt TEXT NOT NULL,
                    seed BIGINT NOT NULL,
                    base_model_checkpoint TEXT NOT NULL,
                    lora_adapter_path TEXT NOT NULL,
                    output_path TEXT NOT NULL,
                    output_object_uri TEXT,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    num_inference_steps INTEGER NOT NULL,
                    guidance_scale DOUBLE PRECISION NOT NULL,
                    device TEXT NOT NULL,
                    latency_ms DOUBLE PRECISION NOT NULL
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS generation_jobs (
                    job_id TEXT PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    started_at TIMESTAMPTZ,
                    finished_at TIMESTAMPTZ,
                    status TEXT NOT NULL,
                    worker_id TEXT,
                    topic TEXT NOT NULL,
                    count_requested INTEGER NOT NULL,
                    count_generated INTEGER NOT NULL DEFAULT 0,
                    preview_limit INTEGER NOT NULL DEFAULT 5,
                    negative_prompt TEXT NOT NULL,
                    base_seed BIGINT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    num_inference_steps INTEGER NOT NULL,
                    guidance_scale DOUBLE PRECISION NOT NULL,
                    base_model_checkpoint TEXT NOT NULL,
                    lora_adapter_path TEXT NOT NULL,
                    manifest_path TEXT,
                    manifest_object_uri TEXT,
                    error TEXT,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                );
                """
            )
            cur.execute("ALTER TABLE generation_requests ADD COLUMN IF NOT EXISTS batch_id TEXT;")
            cur.execute("ALTER TABLE generation_requests ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'single_prompt';")
            cur.execute("ALTER TABLE generation_requests ADD COLUMN IF NOT EXISTS topic TEXT;")
            cur.execute("ALTER TABLE generation_requests ADD COLUMN IF NOT EXISTS output_object_uri TEXT;")
            cur.execute("ALTER TABLE generation_jobs ADD COLUMN IF NOT EXISTS manifest_object_uri TEXT;")
        conn.commit()


def healthcheck_db() -> bool:
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.fetchone()
        return True
    except Exception:
        return False


def insert_prediction(caption, prediction, latency_ms):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO prediction_requests (
                    caption, predicted_label, confidence, probabilities, model_sha, latency_ms
                )
                VALUES (%s, %s, %s, %s::jsonb, %s, %s);
                """,
                (
                    caption,
                    prediction["predicted_label"],
                    prediction["confidence"],
                    json.dumps(prediction["probabilities"], ensure_ascii=False),
                    prediction["model_sha"],
                    latency_ms,
                ),
            )
        conn.commit()


def fetch_recent_predictions(limit: int = 20):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, created_at, caption, predicted_label, confidence, probabilities, model_sha, latency_ms
                FROM prediction_requests
                ORDER BY created_at DESC
                LIMIT %s;
                """,
                (limit,),
            )
            rows = cur.fetchall()

    result = []
    for row in rows:
        result.append(
            {
                "id": row[0],
                "created_at": row[1].isoformat(),
                "caption": row[2],
                "predicted_label": row[3],
                "confidence": float(row[4]),
                "probabilities": row[5],
                "model_sha": row[6],
                "latency_ms": float(row[7]),
            }
        )
    return result


def fetch_synthetic_dataset_runs(limit: int = 20):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    run_id, created_at, source_type, generator, base_model_checkpoint,
                    resolution, num_inference_steps, guidance_scale, base_seed, device,
                    manifest_object_uri, stats_object_uri, preview_object_uri,
                    model_reference_object_uri, metadata
                FROM synthetic_dataset_runs
                ORDER BY created_at DESC
                LIMIT %s;
                """,
                (limit,),
            )
            rows = cur.fetchall()

    return [
        {
            "run_id": row[0],
            "created_at": row[1].isoformat(),
            "source_type": row[2],
            "generator": row[3],
            "base_model_checkpoint": row[4],
            "resolution": row[5],
            "num_inference_steps": row[6],
            "guidance_scale": float(row[7]),
            "base_seed": row[8],
            "device": row[9],
            "manifest_object_uri": row[10],
            "stats_object_uri": row[11],
            "preview_object_uri": row[12],
            "model_reference_object_uri": row[13],
            "metadata": row[14],
        }
        for row in rows
    ]


def fetch_synthetic_dataset_samples(limit: int = 20):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id, run_id, created_at, prompt, domain_tag, source_type,
                    base_model_checkpoint, seed, resolution, num_inference_steps,
                    guidance_scale, quality_score, quality_flag, image_object_uri
                FROM synthetic_dataset_samples
                ORDER BY id
                LIMIT %s;
                """,
                (limit,),
            )
            rows = cur.fetchall()

    return [
        {
            "id": row[0],
            "run_id": row[1],
            "created_at": row[2].isoformat(),
            "prompt": row[3],
            "domain_tag": row[4],
            "source_type": row[5],
            "base_model_checkpoint": row[6],
            "seed": row[7],
            "resolution": row[8],
            "num_inference_steps": row[9],
            "guidance_scale": float(row[10]),
            "quality_score": float(row[11]),
            "quality_flag": row[12],
            "image_object_uri": row[13],
        }
        for row in rows
    ]


def insert_generation(
    prompt,
    negative_prompt,
    result,
    latency_ms,
    batch_id=None,
    topic=None,
    source="single_prompt",
    output_object_uri=None,
):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO generation_requests (
                    batch_id, source, topic, prompt, negative_prompt, seed, base_model_checkpoint, lora_adapter_path,
                    output_path, output_object_uri, width, height, num_inference_steps, guidance_scale, device, latency_ms
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
                """,
                (
                    batch_id,
                    source,
                    topic,
                    prompt,
                    negative_prompt,
                    result.seed,
                    result.model_id,
                    result.lora_adapter_path,
                    result.output_path,
                    output_object_uri,
                    result.width,
                    result.height,
                    result.num_inference_steps,
                    result.guidance_scale,
                    result.device,
                    latency_ms,
                ),
            )
            generation_id = cur.fetchone()[0]
        conn.commit()
    return generation_id


def _job_to_dict(row):
    if row is None:
        return None
    return {
        "job_id": row[0],
        "created_at": row[1].isoformat(),
        "updated_at": row[2].isoformat(),
        "started_at": row[3].isoformat() if row[3] else None,
        "finished_at": row[4].isoformat() if row[4] else None,
        "status": row[5],
        "worker_id": row[6],
        "topic": row[7],
        "count_requested": row[8],
        "count_generated": row[9],
        "preview_limit": row[10],
        "negative_prompt": row[11],
        "base_seed": row[12],
        "width": row[13],
        "height": row[14],
        "num_inference_steps": row[15],
        "guidance_scale": float(row[16]),
        "base_model_checkpoint": row[17],
        "lora_adapter_path": row[18],
        "manifest_path": row[19],
        "manifest_object_uri": row[20],
        "error": row[21],
        "metadata": row[22],
    }


JOB_COLUMNS = """
    job_id, created_at, updated_at, started_at, finished_at, status, worker_id,
    topic, count_requested, count_generated, preview_limit, negative_prompt,
    base_seed, width, height, num_inference_steps, guidance_scale,
    base_model_checkpoint, lora_adapter_path, manifest_path, manifest_object_uri, error, metadata
"""

JOB_COLUMNS_FROM_JOB_ALIAS = """
    job.job_id, job.created_at, job.updated_at, job.started_at, job.finished_at, job.status, job.worker_id,
    job.topic, job.count_requested, job.count_generated, job.preview_limit, job.negative_prompt,
    job.base_seed, job.width, job.height, job.num_inference_steps, job.guidance_scale,
    job.base_model_checkpoint, job.lora_adapter_path, job.manifest_path, job.manifest_object_uri, job.error, job.metadata
"""


def create_generation_job(
    job_id,
    topic,
    count_requested,
    preview_limit,
    negative_prompt,
    base_seed,
    width,
    height,
    num_inference_steps,
    guidance_scale,
    base_model_checkpoint,
    lora_adapter_path,
    metadata=None,
):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO generation_jobs (
                    job_id, status, topic, count_requested, preview_limit, negative_prompt,
                    base_seed, width, height, num_inference_steps, guidance_scale,
                    base_model_checkpoint, lora_adapter_path, metadata
                )
                VALUES (%s, 'queued', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                RETURNING {JOB_COLUMNS};
                """,
                (
                    job_id,
                    topic,
                    count_requested,
                    preview_limit,
                    negative_prompt,
                    base_seed,
                    width,
                    height,
                    num_inference_steps,
                    guidance_scale,
                    base_model_checkpoint,
                    lora_adapter_path,
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )
            row = cur.fetchone()
        conn.commit()
    return _job_to_dict(row)


def acquire_next_generation_job(worker_id):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                WITH next_job AS (
                    SELECT job_id
                    FROM generation_jobs
                    WHERE status = 'queued'
                    ORDER BY created_at
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                )
                UPDATE generation_jobs AS job
                SET status = 'running',
                    worker_id = %s,
                    started_at = COALESCE(started_at, NOW()),
                    updated_at = NOW()
                FROM next_job
                WHERE job.job_id = next_job.job_id
                RETURNING {JOB_COLUMNS_FROM_JOB_ALIAS};
                """,
                (worker_id,),
            )
            row = cur.fetchone()
        conn.commit()
    return _job_to_dict(row)


def update_generation_job_progress(job_id, count_generated):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE generation_jobs
                SET count_generated = %s, updated_at = NOW()
                WHERE job_id = %s;
                """,
                (count_generated, job_id),
            )
        conn.commit()


def complete_generation_job(job_id, manifest_path, manifest_object_uri=None):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE generation_jobs
                SET status = 'done',
                    finished_at = NOW(),
                    updated_at = NOW(),
                    manifest_path = %s,
                    manifest_object_uri = %s,
                    error = NULL
                WHERE job_id = %s
                RETURNING {JOB_COLUMNS};
                """,
                (manifest_path, manifest_object_uri, job_id),
            )
            row = cur.fetchone()
        conn.commit()
    return _job_to_dict(row)


def fail_generation_job(job_id, error):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE generation_jobs
                SET status = 'failed',
                    finished_at = NOW(),
                    updated_at = NOW(),
                    error = %s
                WHERE job_id = %s
                RETURNING {JOB_COLUMNS};
                """,
                (error, job_id),
            )
            row = cur.fetchone()
        conn.commit()
    return _job_to_dict(row)


def fetch_generation_job(job_id):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {JOB_COLUMNS}
                FROM generation_jobs
                WHERE job_id = %s;
                """,
                (job_id,),
            )
            row = cur.fetchone()
    return _job_to_dict(row)


def fetch_recent_generation_jobs(limit=20):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {JOB_COLUMNS}
                FROM generation_jobs
                ORDER BY created_at DESC
                LIMIT %s;
                """,
                (limit,),
            )
            rows = cur.fetchall()
    return [_job_to_dict(row) for row in rows]


def fetch_generations_by_batch(batch_id, limit=1000):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id, created_at, batch_id, source, topic, prompt, negative_prompt,
                    seed, base_model_checkpoint, lora_adapter_path, output_path, output_object_uri, width,
                    height, num_inference_steps, guidance_scale, device, latency_ms
                FROM generation_requests
                WHERE batch_id = %s
                ORDER BY id
                LIMIT %s;
                """,
                (batch_id, limit),
            )
            rows = cur.fetchall()

    return [_generation_to_dict(row) for row in rows]


def fetch_generation_by_id(generation_id):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id, created_at, batch_id, source, topic, prompt, negative_prompt,
                    seed, base_model_checkpoint, lora_adapter_path, output_path, output_object_uri, width,
                    height, num_inference_steps, guidance_scale, device, latency_ms
                FROM generation_requests
                WHERE id = %s;
                """,
                (generation_id,),
            )
            row = cur.fetchone()

    return _generation_to_dict(row) if row else None


def _generation_to_dict(row):
    return {
        "id": row[0],
        "created_at": row[1].isoformat(),
        "batch_id": row[2],
        "source": row[3],
        "topic": row[4],
        "prompt": row[5],
        "negative_prompt": row[6],
        "seed": row[7],
        "base_model_checkpoint": row[8],
        "lora_adapter_path": row[9],
        "output_path": row[10],
        "output_object_uri": row[11],
        "width": row[12],
        "height": row[13],
        "num_inference_steps": row[14],
        "guidance_scale": float(row[15]),
        "device": row[16],
        "latency_ms": float(row[17]),
    }


def fetch_recent_generations(limit: int = 20):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id, created_at, batch_id, source, topic, prompt, negative_prompt,
                    seed, base_model_checkpoint, lora_adapter_path, output_path, output_object_uri, width,
                    height, num_inference_steps, guidance_scale, device, latency_ms
                FROM generation_requests
                ORDER BY created_at DESC
                LIMIT %s;
                """,
                (limit,),
            )
            rows = cur.fetchall()

    return [_generation_to_dict(row) for row in rows]
