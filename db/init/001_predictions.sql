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
