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

CREATE TABLE IF NOT EXISTS generation_requests (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    prompt TEXT NOT NULL,
    negative_prompt TEXT NOT NULL,
    seed BIGINT NOT NULL,
    base_model_checkpoint TEXT NOT NULL,
    lora_adapter_path TEXT NOT NULL,
    output_path TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    num_inference_steps INTEGER NOT NULL,
    guidance_scale DOUBLE PRECISION NOT NULL,
    device TEXT NOT NULL,
    latency_ms DOUBLE PRECISION NOT NULL
);
