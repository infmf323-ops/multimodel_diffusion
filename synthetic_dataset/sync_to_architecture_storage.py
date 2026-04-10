import csv
import io
import json
import os
from pathlib import Path

import boto3
import psycopg
from botocore.client import Config
from botocore.exceptions import ClientError


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "synthetic_manifest.csv"
STATS_PATH = ROOT / "synthetic_stats.json"
PREVIEW_PATH = ROOT / "synthetic_dataset_preview.png"
PLOTS_DIR = ROOT / "plots"

RUN_ID = os.getenv("SYNTHETIC_DATASET_RUN_ID", "diffusion-demo-20260410")
BUCKET = os.getenv("S3_BUCKET", "synthetic-datasets")
S3_PREFIX = os.getenv("S3_PREFIX", f"datasets/{RUN_ID}")
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID", "minioadmin")
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY", "minioadmin")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://multimodal:multimodal@localhost:5432/multimodal_diffusion",
)


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=S3_ACCESS_KEY_ID,
        aws_secret_access_key=S3_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def ensure_bucket(client):
    try:
        client.head_bucket(Bucket=BUCKET)
    except ClientError:
        client.create_bucket(Bucket=BUCKET)


def clear_prefix(client):
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=f"{S3_PREFIX}/"):
        objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
        for start in range(0, len(objects), 1000):
            client.delete_objects(Bucket=BUCKET, Delete={"Objects": objects[start : start + 1000]})


def upload_file(client, local_path: Path, object_key: str, content_type: str):
    client.upload_file(
        str(local_path),
        BUCKET,
        object_key,
        ExtraArgs={"ContentType": content_type},
    )
    return f"s3://{BUCKET}/{object_key}"


def upload_json(client, payload: dict, object_key: str):
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    client.upload_fileobj(
        io.BytesIO(body),
        BUCKET,
        object_key,
        ExtraArgs={"ContentType": "application/json"},
    )
    return f"s3://{BUCKET}/{object_key}"


def read_rows():
    with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def read_stats():
    return json.loads(STATS_PATH.read_text(encoding="utf-8"))


def ensure_tables(conn):
    with conn.cursor() as cur:
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


def upsert_run(conn, stats, manifest_uri, stats_uri, preview_uri, model_ref_uri):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO synthetic_dataset_runs (
                run_id, source_type, generator, base_model_checkpoint, resolution,
                num_inference_steps, guidance_scale, base_seed, device,
                manifest_object_uri, stats_object_uri, preview_object_uri,
                model_reference_object_uri, metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (run_id) DO UPDATE SET
                source_type = EXCLUDED.source_type,
                generator = EXCLUDED.generator,
                base_model_checkpoint = EXCLUDED.base_model_checkpoint,
                resolution = EXCLUDED.resolution,
                num_inference_steps = EXCLUDED.num_inference_steps,
                guidance_scale = EXCLUDED.guidance_scale,
                base_seed = EXCLUDED.base_seed,
                device = EXCLUDED.device,
                manifest_object_uri = EXCLUDED.manifest_object_uri,
                stats_object_uri = EXCLUDED.stats_object_uri,
                preview_object_uri = EXCLUDED.preview_object_uri,
                model_reference_object_uri = EXCLUDED.model_reference_object_uri,
                metadata = EXCLUDED.metadata;
            """,
            (
                RUN_ID,
                stats["source_type"],
                stats["generator"],
                stats["base_model_checkpoint"],
                stats["resolution"],
                int(stats["num_inference_steps"]),
                float(stats["guidance_scale"]),
                int(stats["base_seed"]),
                stats["device"],
                manifest_uri,
                stats_uri,
                preview_uri,
                model_ref_uri,
                json.dumps(stats, ensure_ascii=False),
            ),
        )
    conn.commit()


def upsert_sample(conn, row, image_uri):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO synthetic_dataset_samples (
                id, run_id, prompt, generation_prompt, negative_prompt, domain_tag,
                source_type, generator, base_model_checkpoint, seed, resolution,
                num_inference_steps, guidance_scale, quality_score, quality_score_type,
                quality_flag, local_image_path, image_object_uri, metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (id) DO UPDATE SET
                run_id = EXCLUDED.run_id,
                prompt = EXCLUDED.prompt,
                generation_prompt = EXCLUDED.generation_prompt,
                negative_prompt = EXCLUDED.negative_prompt,
                domain_tag = EXCLUDED.domain_tag,
                source_type = EXCLUDED.source_type,
                generator = EXCLUDED.generator,
                base_model_checkpoint = EXCLUDED.base_model_checkpoint,
                seed = EXCLUDED.seed,
                resolution = EXCLUDED.resolution,
                num_inference_steps = EXCLUDED.num_inference_steps,
                guidance_scale = EXCLUDED.guidance_scale,
                quality_score = EXCLUDED.quality_score,
                quality_score_type = EXCLUDED.quality_score_type,
                quality_flag = EXCLUDED.quality_flag,
                local_image_path = EXCLUDED.local_image_path,
                image_object_uri = EXCLUDED.image_object_uri,
                metadata = EXCLUDED.metadata;
            """,
            (
                row["id"],
                RUN_ID,
                row["prompt"],
                row["generation_prompt"],
                row["negative_prompt"],
                row["domain_tag"],
                row["source_type"],
                row["generator"],
                row["base_model_checkpoint"],
                int(row["seed"]),
                row["resolution"],
                int(row["num_inference_steps"]),
                float(row["guidance_scale"]),
                float(row["quality_score"]),
                row["quality_score_type"],
                row["quality_flag"],
                row["image_path"],
                image_uri,
                json.dumps(row, ensure_ascii=False),
            ),
        )


def main():
    rows = read_rows()
    stats = read_stats()
    client = s3_client()
    ensure_bucket(client)
    clear_prefix(client)

    manifest_uri = upload_file(client, MANIFEST_PATH, f"{S3_PREFIX}/manifest/synthetic_manifest.csv", "text/csv")
    stats_uri = upload_file(client, STATS_PATH, f"{S3_PREFIX}/metadata/synthetic_stats.json", "application/json")
    preview_uri = upload_file(client, PREVIEW_PATH, f"{S3_PREFIX}/previews/synthetic_dataset_preview.png", "image/png")
    upload_file(
        client,
        PLOTS_DIR / "synthetic_domain_distribution.png",
        f"{S3_PREFIX}/plots/synthetic_domain_distribution.png",
        "image/png",
    )

    model_ref = {
        "model_id": stats["base_model_checkpoint"],
        "generator": stats["generator"],
        "note": "For local demo we store the reproducible Hugging Face model reference instead of copying checkpoint weights to MinIO.",
        "local_cache_hint": "hf_cache/",
    }
    model_ref_uri = upload_json(client, model_ref, f"{S3_PREFIX}/model_refs/segmind_tiny_sd.json")

    image_uris = {}
    for row in rows:
        image_path = ROOT / row["image_path"]
        object_key = f"{S3_PREFIX}/images/{image_path.name}"
        image_uris[row["id"]] = upload_file(client, image_path, object_key, "image/png")

    with psycopg.connect(DATABASE_URL) as conn:
        ensure_tables(conn)
        upsert_run(conn, stats, manifest_uri, stats_uri, preview_uri, model_ref_uri)
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM synthetic_dataset_samples WHERE run_id = %s AND id <> ALL(%s);",
                (RUN_ID, [row["id"] for row in rows]),
            )
        for row in rows:
            upsert_sample(conn, row, image_uris[row["id"]])
        conn.commit()

    print(
        json.dumps(
            {
                "run_id": RUN_ID,
                "bucket": BUCKET,
                "s3_prefix": S3_PREFIX,
                "samples_imported": len(rows),
                "manifest_object_uri": manifest_uri,
                "stats_object_uri": stats_uri,
                "preview_object_uri": preview_uri,
                "model_reference_object_uri": model_ref_uri,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
