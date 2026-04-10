import csv
import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageStat

try:
    import mlflow
except ImportError:
    mlflow = None

from generate_diffusion_dataset import NEGATIVE_PROMPT, PROMPTS, enhanced_prompt


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
CACHE_DIR = REPO_ROOT / "hf_cache"
LOCAL_MODEL_SNAPSHOTS = CACHE_DIR / "hub" / "models--segmind--tiny-sd" / "snapshots"
OUTPUT_ROOT = ROOT / "parameter_experiments"
SUMMARY_JSON = OUTPUT_ROOT / "summary.json"
SUMMARY_CSV = OUTPUT_ROOT / "summary.csv"
COMPARISON_PLOT = OUTPUT_ROOT / "experiment_metric_comparison.png"
PREVIEW_PLOT = OUTPUT_ROOT / "experiment_preview_grid.png"
PROMPT_POOL_JSON = OUTPUT_ROOT / "prompt_pool.json"
MLFLOW_ENABLED = os.getenv("MLFLOW_ENABLED", "true").lower() == "true"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "synthetic-dataset-parameter-search")


@dataclass
class ExperimentConfig:
    name: str
    width: int
    height: int
    num_inference_steps: int
    guidance_scale: float
    quality_threshold: float = 0.60


EXPERIMENTS = [
    ExperimentConfig("fast_256", width=256, height=256, num_inference_steps=10, guidance_scale=4.0),
    ExperimentConfig("balanced_256", width=256, height=256, num_inference_steps=14, guidance_scale=5.5),
    ExperimentConfig("detailed_256", width=256, height=256, num_inference_steps=20, guidance_scale=7.0),
    ExperimentConfig("balanced_320", width=320, height=320, num_inference_steps=16, guidance_scale=5.5),
    ExperimentConfig("guided_320", width=320, height=320, num_inference_steps=22, guidance_scale=7.5),
]


def load_font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


TITLE_FONT = load_font(24)
TEXT_FONT = load_font(16)
SMALL_FONT = load_font(14)


def select_balanced_prompts(per_domain: int = 4):
    grouped = defaultdict(list)
    for domain, prompt in PROMPTS:
        if len(grouped[domain]) < per_domain:
            grouped[domain].append((domain, prompt))

    selected = []
    for domain in ["sports", "animals_nature", "people_entertainment", "places_objects"]:
        selected.extend(grouped[domain])
    return selected


def average_hash(image: Image.Image, hash_size: int = 8) -> np.ndarray:
    gray = image.convert("L").resize((hash_size, hash_size), Image.Resampling.BILINEAR)
    arr = np.asarray(gray, dtype=np.float32)
    return (arr >= arr.mean()).astype(np.uint8).flatten()


def image_metrics(image: Image.Image) -> dict:
    rgb = image.convert("RGB")
    gray = image.convert("L")
    stat_rgb = ImageStat.Stat(rgb)
    stat_gray = ImageStat.Stat(gray)

    contrast = min(stat_gray.stddev[0] / 64.0, 1.0)
    arr = np.asarray(rgb).astype(np.float32) / 255.0
    channel_max = arr.max(axis=2)
    channel_min = arr.min(axis=2)
    saturation = float(np.mean((channel_max - channel_min) / np.maximum(channel_max, 1e-6)))
    color_variation = min(float(np.mean(stat_rgb.stddev)) / 72.0, 1.0)
    brightness = float(np.mean(channel_max))

    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_density = min(ImageStat.Stat(edges).mean[0] / 42.0, 1.0)
    brightness_penalty = abs(brightness - 0.55) / 0.55

    quality_score = 0.35 * contrast + 0.25 * saturation + 0.20 * color_variation + 0.20 * edge_density
    quality_score = max(0.0, min(1.0, quality_score * (1.0 - 0.30 * brightness_penalty)))

    return {
        "contrast": round(float(contrast), 4),
        "saturation": round(float(saturation), 4),
        "color_variation": round(float(color_variation), 4),
        "edge_density": round(float(edge_density), 4),
        "brightness": round(float(brightness), 4),
        "quality_score": round(float(quality_score), 4),
    }


def dataset_metrics(rows: list[dict], quality_threshold: float) -> dict:
    if not rows:
        return {}

    quality_scores = np.array([row["quality_score"] for row in rows], dtype=np.float32)
    contrast = np.array([row["contrast"] for row in rows], dtype=np.float32)
    saturation = np.array([row["saturation"] for row in rows], dtype=np.float32)
    edge_density = np.array([row["edge_density"] for row in rows], dtype=np.float32)
    color_variation = np.array([row["color_variation"] for row in rows], dtype=np.float32)
    latency_ms = np.array([row["latency_ms"] for row in rows], dtype=np.float32)
    file_sizes_kb = np.array([row["file_size_kb"] for row in rows], dtype=np.float32)
    hash_vectors = [row["ahash"] for row in rows]

    pair_distances = []
    duplicate_pairs = 0
    for left, right in combinations(hash_vectors, 2):
        distance = int(np.count_nonzero(left != right))
        pair_distances.append(distance / len(left))
        if distance <= 5:
            duplicate_pairs += 1

    domain_counts = Counter(row["domain_tag"] for row in rows)
    probabilities = np.array(list(domain_counts.values()), dtype=np.float32) / len(rows)
    if len(probabilities) > 1:
        entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-8)))
        balance_score = entropy / math.log(len(probabilities))
    else:
        balance_score = 0.0

    total_pairs = max(len(pair_distances), 1)
    total_time_s = float(np.sum(latency_ms) / 1000.0)
    return {
        "sample_count": len(rows),
        "avg_quality_score": round(float(np.mean(quality_scores)), 4),
        "median_quality_score": round(float(np.median(quality_scores)), 4),
        "min_quality_score": round(float(np.min(quality_scores)), 4),
        "max_quality_score": round(float(np.max(quality_scores)), 4),
        "high_quality_share": round(float(np.mean(quality_scores >= quality_threshold)), 4),
        "avg_contrast": round(float(np.mean(contrast)), 4),
        "avg_saturation": round(float(np.mean(saturation)), 4),
        "avg_color_variation": round(float(np.mean(color_variation)), 4),
        "avg_edge_density": round(float(np.mean(edge_density)), 4),
        "diversity_score": round(float(np.mean(pair_distances)) if pair_distances else 0.0, 4),
        "near_duplicate_rate": round(float(duplicate_pairs / total_pairs), 4),
        "domain_balance_score": round(balance_score, 4),
        "avg_latency_ms": round(float(np.mean(latency_ms)), 2),
        "total_generation_time_s": round(total_time_s, 2),
        "images_per_minute": round(float((len(rows) / total_time_s) * 60.0), 2) if total_time_s > 0 else 0.0,
        "avg_file_size_kb": round(float(np.mean(file_sizes_kb)), 2),
        "quality_threshold": quality_threshold,
        "hash_type": "average_hash_8x8",
        "near_duplicate_hamming_threshold": 5,
        "domain_distribution": dict(domain_counts),
    }


def write_manifest(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "domain_tag",
        "prompt",
        "generation_prompt",
        "negative_prompt",
        "image_path",
        "seed",
        "resolution",
        "num_inference_steps",
        "guidance_scale",
        "latency_ms",
        "file_size_kb",
        "contrast",
        "saturation",
        "color_variation",
        "edge_density",
        "brightness",
        "quality_score",
        "quality_flag",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            clean_row = {key: value for key, value in row.items() if key in fieldnames}
            writer.writerow(clean_row)


def write_summary_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def configure_mlflow() -> bool:
    if not MLFLOW_ENABLED or mlflow is None:
        return False
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        return True
    except Exception as exc:
        print(f"[mlflow] disabled: {exc}")
        return False


def log_experiment_run_to_mlflow(config: ExperimentConfig, stats: dict, exp_dir: Path):
    if mlflow is None:
        return
    with mlflow.start_run(run_name=config.name, nested=True):
        mlflow.set_tags(
            {
                "lab": "LR3",
                "task": "synthetic_dataset_parameter_search",
                "experiment_type": "dataset_generation",
                "base_model_checkpoint": stats["base_model_checkpoint"],
            }
        )
        mlflow.log_params(
            {
                "experiment_name": config.name,
                "width": config.width,
                "height": config.height,
                "resolution": f"{config.width}x{config.height}",
                "num_inference_steps": config.num_inference_steps,
                "guidance_scale": config.guidance_scale,
                "quality_threshold": config.quality_threshold,
                "prompt_count": stats["prompt_count"],
                "tracking_mode": "dataset_metrics",
            }
        )
        mlflow.log_metrics(
            {
                "avg_quality_score": float(stats["avg_quality_score"]),
                "median_quality_score": float(stats["median_quality_score"]),
                "min_quality_score": float(stats["min_quality_score"]),
                "max_quality_score": float(stats["max_quality_score"]),
                "high_quality_share": float(stats["high_quality_share"]),
                "avg_contrast": float(stats["avg_contrast"]),
                "avg_saturation": float(stats["avg_saturation"]),
                "avg_color_variation": float(stats["avg_color_variation"]),
                "avg_edge_density": float(stats["avg_edge_density"]),
                "diversity_score": float(stats["diversity_score"]),
                "near_duplicate_rate": float(stats["near_duplicate_rate"]),
                "domain_balance_score": float(stats["domain_balance_score"]),
                "avg_latency_ms": float(stats["avg_latency_ms"]),
                "total_generation_time_s": float(stats["total_generation_time_s"]),
                "images_per_minute": float(stats["images_per_minute"]),
                "avg_file_size_kb": float(stats["avg_file_size_kb"]),
            }
        )
        mlflow.log_dict(stats, "stats.json")
        mlflow.log_artifacts(str(exp_dir), artifact_path=f"parameter_experiments/{config.name}")


def metric_bar_panel(draw: ImageDraw.ImageDraw, area: tuple[int, int, int, int], title: str, rows: list[dict], key: str, color: str):
    x0, y0, x1, y1 = area
    draw.rounded_rectangle(area, radius=18, outline="#d1d5db", width=2, fill="#ffffff")
    draw.text((x0 + 20, y0 + 16), title, fill="#111827", font=TEXT_FONT)
    values = [float(row[key]) for row in rows]
    max_value = max(values) if values else 1.0
    bar_area_top = y0 + 70
    bar_area_bottom = y1 - 46
    bar_width = max(36, int((x1 - x0 - 80) / max(len(rows), 1)))
    gap = 18

    for index, row in enumerate(rows):
        value = float(row[key])
        scaled_height = 0 if max_value == 0 else int((value / max_value) * (bar_area_bottom - bar_area_top))
        left = x0 + 28 + index * (bar_width + gap)
        top = bar_area_bottom - scaled_height
        right = left + bar_width
        draw.rounded_rectangle((left, top, right, bar_area_bottom), radius=10, fill=color)
        draw.text((left, top - 22), f"{value:.3f}", fill="#374151", font=SMALL_FONT)
        draw.text((left - 4, bar_area_bottom + 8), row["experiment"], fill="#374151", font=SMALL_FONT)


def build_comparison_plot(summary_rows: list[dict]):
    canvas = Image.new("RGB", (1680, 980), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 28), "Dataset-level comparison of synthetic generation parameters", fill="#111827", font=TITLE_FONT)
    metric_bar_panel(draw, (40, 90, 800, 450), "Average quality score", summary_rows, "avg_quality_score", "#2563eb")
    metric_bar_panel(draw, (840, 90, 1600, 450), "Diversity score", summary_rows, "diversity_score", "#16a34a")
    metric_bar_panel(draw, (40, 500, 800, 860), "Average latency (ms)", summary_rows, "avg_latency_ms", "#f97316")
    metric_bar_panel(draw, (840, 500, 1600, 860), "High-quality share", summary_rows, "high_quality_share", "#7c3aed")
    draw.text(
        (40, 900),
        "Quality and diversity are dataset metrics; lower duplicate rate and lower latency are also inspected separately in summary.json / summary.csv.",
        fill="#374151",
        font=TEXT_FONT,
    )
    canvas.save(COMPARISON_PLOT)


def build_preview_plot(summary_rows: list[dict], experiment_rows: dict[str, list[dict]]):
    card_width = 290
    card_height = 370
    cols = len(summary_rows)
    canvas = Image.new("RGB", (cols * card_width + 40, card_height + 80), "#f3f4f6")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 20), "Representative preview by experiment", fill="#111827", font=TITLE_FONT)

    for index, summary in enumerate(summary_rows):
        rows = sorted(experiment_rows[summary["experiment"]], key=lambda item: item["quality_score"], reverse=True)
        best = rows[0]
        image = Image.open(best["absolute_image_path"]).convert("RGB").resize((250, 250))
        x = 20 + index * card_width
        y = 70
        draw.rounded_rectangle((x, y, x + 270, y + 320), radius=18, fill="#ffffff", outline="#d1d5db", width=2)
        canvas.paste(image, (x + 10, y + 10))
        draw.text((x + 12, y + 272), summary["experiment"], fill="#111827", font=TEXT_FONT)
        draw.text((x + 12, y + 298), f"q={best['quality_score']:.3f}, latency={best['latency_ms']:.1f} ms", fill="#374151", font=SMALL_FONT)
        draw.text((x + 12, y + 318), best["domain_tag"], fill="#6b7280", font=SMALL_FONT)
    canvas.save(PREVIEW_PLOT)


def load_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model_source = "segmind/tiny-sd"
    if LOCAL_MODEL_SNAPSHOTS.exists():
        snapshots = sorted([path for path in LOCAL_MODEL_SNAPSHOTS.iterdir() if path.is_dir()])
        if snapshots:
            model_source = str(snapshots[-1])
    pipe = StableDiffusionPipeline.from_pretrained(
        model_source,
        cache_dir=str(CACHE_DIR),
        local_files_only=True,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    return pipe, device


def run():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    selected_prompts = select_balanced_prompts(per_domain=4)
    PROMPT_POOL_JSON.write_text(
        json.dumps(
            [
                {"domain_tag": domain, "prompt": prompt, "generation_prompt": enhanced_prompt(prompt)}
                for domain, prompt in selected_prompts
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    pipe, device = load_pipeline()
    summary_rows = []
    experiment_rows = {}
    base_seed = 20260411
    mlflow_active = configure_mlflow()
    parent_run = None

    if mlflow_active and mlflow is not None:
        parent_run = mlflow.start_run(run_name=f"dataset-sweep-{base_seed}")
        mlflow.set_tags(
            {
                "lab": "LR3",
                "task": "synthetic_dataset_parameter_search",
                "project": "multimodal_diffusion",
                "base_model_checkpoint": "segmind/tiny-sd",
            }
        )
        mlflow.log_params(
            {
                "selected_prompt_count": len(selected_prompts),
                "domain_count": 4,
                "base_model_checkpoint": "segmind/tiny-sd",
                "base_seed": base_seed,
                "tracking_uri": MLFLOW_TRACKING_URI,
            }
        )
        mlflow.log_artifact(str(PROMPT_POOL_JSON), artifact_path="dataset_setup")

    try:
        for exp_index, config in enumerate(EXPERIMENTS):
            exp_dir = OUTPUT_ROOT / config.name
            image_dir = exp_dir / "images"
            image_dir.mkdir(parents=True, exist_ok=True)
            for stale_image in image_dir.glob("*.png"):
                stale_image.unlink()

            rows = []
            print(f"[experiment] {config.name} -> {config.width}x{config.height}, steps={config.num_inference_steps}, guidance={config.guidance_scale}")
            for prompt_index, (domain_tag, prompt) in enumerate(selected_prompts, start=1):
                seed = base_seed + exp_index * 1000 + prompt_index
                generator = torch.Generator(device=device).manual_seed(seed)
                generation_prompt = enhanced_prompt(prompt)
                item_started = time.perf_counter()
                image = pipe(
                    generation_prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    width=config.width,
                    height=config.height,
                    num_inference_steps=config.num_inference_steps,
                    guidance_scale=config.guidance_scale,
                    generator=generator,
                ).images[0]
                latency_ms = (time.perf_counter() - item_started) * 1000.0

                image_name = f"{config.name}_{prompt_index:03d}_{domain_tag}.png"
                image_path = image_dir / image_name
                image.save(image_path)

                metrics = image_metrics(image)
                quality_flag = "good" if metrics["quality_score"] >= config.quality_threshold else "review"
                row = {
                    "id": f"{config.name}_{prompt_index:03d}",
                    "domain_tag": domain_tag,
                    "prompt": prompt,
                    "generation_prompt": generation_prompt,
                    "negative_prompt": NEGATIVE_PROMPT,
                    "image_path": str(image_path.relative_to(ROOT)),
                    "absolute_image_path": str(image_path),
                    "seed": seed,
                    "resolution": f"{config.width}x{config.height}",
                    "num_inference_steps": config.num_inference_steps,
                    "guidance_scale": config.guidance_scale,
                    "latency_ms": round(latency_ms, 2),
                    "file_size_kb": round(image_path.stat().st_size / 1024.0, 2),
                    "quality_flag": quality_flag,
                    "ahash": average_hash(image),
                    **metrics,
                }
                rows.append(row)
                print(f"  - {row['id']} q={row['quality_score']:.3f} latency={row['latency_ms']:.1f}ms")

            experiment_rows[config.name] = rows
            stats = dataset_metrics(rows, config.quality_threshold)
            stats.update(
                {
                    "experiment": config.name,
                    "device": device,
                    "base_model_checkpoint": "segmind/tiny-sd",
                    "resolution": f"{config.width}x{config.height}",
                    "num_inference_steps": config.num_inference_steps,
                    "guidance_scale": config.guidance_scale,
                    "quality_threshold": config.quality_threshold,
                    "prompt_count": len(selected_prompts),
                }
            )

            manifest_path = exp_dir / "manifest.csv"
            stats_path = exp_dir / "stats.json"
            write_manifest(manifest_path, rows)
            stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
            summary_rows.append(stats)

            if mlflow_active and mlflow is not None:
                log_experiment_run_to_mlflow(config, stats, exp_dir)

        summary_rows.sort(key=lambda item: item["avg_quality_score"], reverse=True)
        write_summary_csv(SUMMARY_CSV, summary_rows)
        SUMMARY_JSON.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        build_comparison_plot(summary_rows)
        build_preview_plot(summary_rows, experiment_rows)

        if mlflow_active and mlflow is not None and parent_run is not None and summary_rows:
            best = summary_rows[0]
            mlflow.log_metrics(
                {
                    "best_avg_quality_score": float(best["avg_quality_score"]),
                    "best_diversity_score": float(best["diversity_score"]),
                    "best_high_quality_share": float(best["high_quality_share"]),
                    "best_avg_latency_ms": float(best["avg_latency_ms"]),
                }
            )
            mlflow.log_params(
                {
                    "best_experiment": best["experiment"],
                    "best_resolution": best["resolution"],
                    "best_guidance_scale": best["guidance_scale"],
                    "best_num_inference_steps": best["num_inference_steps"],
                }
            )
            mlflow.log_artifact(str(SUMMARY_JSON), artifact_path="dataset_summary")
            mlflow.log_artifact(str(SUMMARY_CSV), artifact_path="dataset_summary")
            mlflow.log_artifact(str(COMPARISON_PLOT), artifact_path="plots")
            mlflow.log_artifact(str(PREVIEW_PLOT), artifact_path="plots")

        print(json.dumps(summary_rows, ensure_ascii=False, indent=2))
    finally:
        if parent_run is not None and mlflow is not None:
            mlflow.end_run()


if __name__ == "__main__":
    run()
