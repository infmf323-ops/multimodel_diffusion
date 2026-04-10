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
