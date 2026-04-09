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
