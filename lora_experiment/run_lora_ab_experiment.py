import argparse
import csv
import gc
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, StableDiffusionPipeline
from peft import LoraConfig, get_peft_model_state_dict
from PIL import Image, ImageDraw, ImageFilter, ImageStat
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MODEL = "segmind/tiny-sd"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "lora_experiment" / "artifacts"
DEFAULT_SYNTHETIC_MANIFEST = REPO_ROOT / "synthetic_dataset" / "synthetic_manifest.csv"
DEFAULT_HF_HOME = REPO_ROOT / "hf_cache"


@dataclass
class ExperimentConfig:
    base_model: str
    output_dir: str
    synthetic_manifest: str
    seed_manifest: str | None
    allow_proxy_seed: bool
    split_seed: int
    eval_seed: int
    resolution: int
    train_steps: int
    eval_steps: int
    batch_size: int
    learning_rate: float
    lora_rank: int
    lora_alpha: int
    guidance_scale: float
    max_eval_prompts: int
    clip_model: str
    no_clip: bool
    skip_training: bool
    proxy_seed_train_size: int
    proxy_val_size: int
    proxy_test_size: int


class TextImageDataset(Dataset):
    def __init__(self, rows, resolution):
        self.rows = rows
        self.resolution = resolution

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = Image.open(row["resolved_image_path"]).convert("RGB")
        image = image.resize((self.resolution, self.resolution), Image.Resampling.BICUBIC)
        array = np.asarray(image).astype(np.float32) / 127.5 - 1.0
        pixel_values = torch.from_numpy(array).permute(2, 0, 1)
        return {
            "pixel_values": pixel_values,
            "prompt": row["prompt"],
            "id": row["id"],
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Run seed-only vs seed+synthetic LoRA A/B experiment.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--synthetic-manifest", default=str(DEFAULT_SYNTHETIC_MANIFEST))
    parser.add_argument("--seed-manifest", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--allow-proxy-seed", action="store_true")
    parser.add_argument("--split-seed", type=int, default=20260410)
    parser.add_argument("--eval-seed", type=int, default=20260420)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--train-steps", type=int, default=30)
    parser.add_argument("--eval-steps", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--max-eval-prompts", type=int, default=6)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--no-clip", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--proxy-seed-train-size", type=int, default=24)
    parser.add_argument("--proxy-val-size", type=int, default=8)
    parser.add_argument("--proxy-test-size", type=int, default=8)
    return parser.parse_args()


def read_manifest(path):
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = [dict(row) for row in reader]

    manifest_dir = path.parent
    normalized = []
    for row in rows:
        prompt = row.get("prompt") or row.get("caption") or row.get("generation_prompt")
        image_path = row.get("image_path") or row.get("path") or row.get("file_name")
        if not prompt or not image_path:
            continue
        resolved = Path(image_path)
        if not resolved.is_absolute():
            resolved = manifest_dir / resolved
        if resolved.exists():
            item = dict(row)
            item["prompt"] = prompt
            item["image_path"] = image_path
            item["resolved_image_path"] = str(resolved)
            item["id"] = item.get("id") or resolved.stem
            normalized.append(item)
    return normalized


def write_manifest(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "prompt",
        "image_path",
        "resolved_image_path",
        "domain_tag",
        "source_type",
        "split",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def prepare_manifests(config):
    synthetic_rows = read_manifest(config.synthetic_manifest)
    if not synthetic_rows:
        raise RuntimeError(f"No usable synthetic rows found in {config.synthetic_manifest}")

    rng = random.Random(config.split_seed)
    output_dir = Path(config.output_dir)
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    proxy_mode = False
    if config.seed_manifest:
        seed_rows = read_manifest(config.seed_manifest)
        if not seed_rows:
            raise RuntimeError(f"Seed manifest has no usable image+text pairs: {config.seed_manifest}")
        rng.shuffle(seed_rows)
        n = len(seed_rows)
        n_val = max(1, min(config.proxy_val_size, n // 5))
        n_test = max(1, min(config.proxy_test_size, n // 5))
        val_rows = seed_rows[:n_val]
        test_rows = seed_rows[n_val:n_val + n_test]
        seed_train = seed_rows[n_val + n_test:]
        synthetic_train = synthetic_rows
    elif config.allow_proxy_seed:
        proxy_mode = True
        rows = list(synthetic_rows)
        rng.shuffle(rows)
        min_required = config.proxy_seed_train_size + config.proxy_val_size + config.proxy_test_size + 1
        if len(rows) < min_required:
            raise RuntimeError(f"Need at least {min_required} rows for proxy split, got {len(rows)}")
        seed_train = rows[:config.proxy_seed_train_size]
        val_start = config.proxy_seed_train_size
        test_start = val_start + config.proxy_val_size
        synth_start = test_start + config.proxy_test_size
        val_rows = rows[val_start:test_start]
        test_rows = rows[test_start:synth_start]
        synthetic_train = rows[synth_start:]
    else:
        raise RuntimeError(
            "No seed image manifest was provided. Pass --seed-manifest path/to/manifest.csv "
            "or use --allow-proxy-seed for a local demo split based on synthetic_dataset."
        )

    for split_name, rows in [
        ("seed_train", seed_train),
        ("synthetic_train", synthetic_train),
        ("val", val_rows),
        ("test", test_rows),
    ]:
        for row in rows:
            row["split"] = split_name

    seed_plus_synthetic = seed_train + synthetic_train
    write_manifest(manifest_dir / "seed_train.csv", seed_train)
    write_manifest(manifest_dir / "synthetic_train.csv", synthetic_train)
    write_manifest(manifest_dir / "seed_plus_synthetic_train.csv", seed_plus_synthetic)
    write_manifest(manifest_dir / "val.csv", val_rows)
    write_manifest(manifest_dir / "test.csv", test_rows)

    return {
        "proxy_mode": proxy_mode,
        "seed_train": seed_train,
        "seed_plus_synthetic_train": seed_plus_synthetic,
        "synthetic_train": synthetic_train,
        "val": val_rows,
        "test": test_rows,
        "manifest_dir": str(manifest_dir),
    }


def load_training_pipeline(config, device):
    pipe = StableDiffusionPipeline.from_pretrained(
        config.base_model,
        local_files_only=True,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.unet.add_adapter(
        LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
    )
    pipe.vae.to(device=device, dtype=torch.float32)
    pipe.text_encoder.to(device=device, dtype=torch.float32)
    pipe.unet.to(device=device, dtype=torch.float32)
    pipe.unet.train()
    return pipe


def tokenize_prompts(tokenizer, prompts, device):
    tokens = tokenizer(
        prompts,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return tokens.input_ids.to(device)


def compute_denoising_loss(pipe, noise_scheduler, batch, device):
    pixel_values = batch["pixel_values"].to(device=device, dtype=pipe.vae.dtype)
    input_ids = tokenize_prompts(pipe.tokenizer, batch["prompt"], device)
    latents = pipe.vae.encode(pixel_values).latent_dist.sample()
    latents = latents * pipe.vae.config.scaling_factor
    noise = torch.randn_like(latents)
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (latents.shape[0],),
        device=device,
        dtype=torch.long,
    )
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    encoder_hidden_states = pipe.text_encoder(input_ids)[0]
    model_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states).sample

    if noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        target = noise
    return F.mse_loss(model_pred.float(), target.float(), reduction="mean")


@torch.no_grad()
def evaluate_val_loss(pipe, noise_scheduler, rows, config, device, max_batches=4):
    if not rows:
        return None
    dataset = TextImageDataset(rows[:max_batches], config.resolution)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    losses = []
    pipe.unet.eval()
    for batch in loader:
        loss = compute_denoising_loss(pipe, noise_scheduler, batch, device)
        losses.append(float(loss.detach().cpu()))
    pipe.unet.train()
    return round(float(np.mean(losses)), 6) if losses else None


def train_lora(run_name, train_rows, val_rows, config, device):
    run_dir = Path(config.output_dir) / "runs" / run_name
    adapter_dir = run_dir / "adapters"
    run_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_training_pipeline(config, device)
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    dataset = TextImageDataset(train_rows, config.resolution)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    trainable_params = [p for p in pipe.unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)

    loss_log = []
    step = 0
    start = time.perf_counter()

    while step < config.train_steps:
        for batch in loader:
            if step >= config.train_steps:
                break
            optimizer.zero_grad(set_to_none=True)
            loss = compute_denoising_loss(pipe, noise_scheduler, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            step += 1
            loss_value = float(loss.detach().cpu())
            loss_log.append({"step": step, "train_loss": round(loss_value, 6)})
            print(f"{run_name} step {step}/{config.train_steps} train_loss={loss_value:.6f}", flush=True)

    val_loss = evaluate_val_loss(pipe, noise_scheduler, val_rows, config, device)
    elapsed_seconds = round(time.perf_counter() - start, 3)
    lora_state_dict = get_peft_model_state_dict(pipe.unet)
    pipe.save_lora_weights(str(adapter_dir), unet_lora_layers=lora_state_dict)

    train_summary = {
        "run_name": run_name,
        "base_model": config.base_model,
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "train_steps": config.train_steps,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "final_train_loss": loss_log[-1]["train_loss"] if loss_log else None,
        "mean_train_loss": round(float(np.mean([x["train_loss"] for x in loss_log])), 6) if loss_log else None,
        "val_denoising_loss": val_loss,
        "elapsed_seconds": elapsed_seconds,
        "adapter_dir": str(adapter_dir),
    }
    write_json(run_dir / "train_summary.json", train_summary)
    write_jsonl(run_dir / "train_loss.jsonl", loss_log)

    del pipe, optimizer, loader, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return train_summary


def load_generation_pipeline(config, device, adapter_dir=None):
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        config.base_model,
        torch_dtype=dtype,
        local_files_only=True,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.to(device)
    if adapter_dir:
        pipe.load_lora_weights(str(adapter_dir), adapter_name="default")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_eval_images(run_name, adapter_dir, test_rows, config, device):
    out_dir = Path(config.output_dir) / "eval" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe = load_generation_pipeline(config, device, adapter_dir)
    rows = test_rows[:config.max_eval_prompts]
    generated = []
    latencies = []
    for index, row in enumerate(rows):
        generator = torch.Generator(device=device).manual_seed(config.eval_seed + index)
        start = time.perf_counter()
        image = pipe(
            row["prompt"],
            width=config.resolution,
            height=config.resolution,
            num_inference_steps=config.eval_steps,
            guidance_scale=config.guidance_scale,
            generator=generator,
        ).images[0]
        latency = time.perf_counter() - start
        latencies.append(latency)
        image_path = out_dir / f"{index + 1:03d}_{row['id']}.png"
        image.save(image_path)
        generated.append(
            {
                "id": row["id"],
                "prompt": row["prompt"],
                "reference_image_path": row["resolved_image_path"],
                "generated_image_path": str(image_path),
                "seed": config.eval_seed + index,
                "latency_seconds": round(latency, 4),
                "technical_image_score": technical_quality_score(image),
            }
        )
    write_json(out_dir / "generated_manifest.json", generated)
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return generated, round(float(np.mean(latencies)), 4) if latencies else None


def technical_quality_score(image):
    rgb = image.convert("RGB")
    gray = image.convert("L")
    stat = ImageStat.Stat(gray)
    contrast = min(stat.stddev[0] / 64.0, 1.0)
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_stat = ImageStat.Stat(edges)
    sharpness = min(edge_stat.mean[0] / 32.0, 1.0)
    saturation = min(np.mean(ImageStat.Stat(rgb).stddev) / 80.0, 1.0)
    return round(float(0.4 * contrast + 0.4 * sharpness + 0.2 * saturation), 3)


def try_load_clip(model_id, device):
    try:
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained(model_id, local_files_only=True)
        processor = CLIPProcessor.from_pretrained(model_id, local_files_only=True)
        model.to(device)
        model.eval()
        return model, processor, None
    except Exception as exc:
        return None, None, str(exc)


@torch.no_grad()
def clip_text_image_scores(model, processor, rows, image_key, device):
    scores = []
    for row in rows:
        image = Image.open(row[image_key]).convert("RGB")
        inputs = processor(text=[row["prompt"]], images=[image], return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model(**inputs)
        image_embeds = F.normalize(outputs.image_embeds.float(), dim=-1)
        text_embeds = F.normalize(outputs.text_embeds.float(), dim=-1)
        scores.append(float((image_embeds * text_embeds).sum(dim=-1).item()))
    return scores


@torch.no_grad()
def clip_image_embeddings(model, processor, image_paths, device):
    embeddings = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        inputs = processor(images=[image], return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        image_features = model.get_image_features(**inputs)
        if not torch.is_tensor(image_features):
            image_features = image_features.image_embeds if hasattr(image_features, "image_embeds") else image_features.pooler_output
        embeddings.append(F.normalize(image_features.float(), dim=-1).cpu().numpy()[0])
    return np.asarray(embeddings, dtype=np.float32)


def polynomial_mmd2(x, y):
    if len(x) < 2 or len(y) < 2:
        return None
    dim = x.shape[1]
    k_xx = (x @ x.T / dim + 1.0) ** 3
    k_yy = (y @ y.T / dim + 1.0) ** 3
    k_xy = (x @ y.T / dim + 1.0) ** 3
    np.fill_diagonal(k_xx, 0.0)
    np.fill_diagonal(k_yy, 0.0)
    m = len(x)
    n = len(y)
    return float(k_xx.sum() / (m * (m - 1)) + k_yy.sum() / (n * (n - 1)) - 2.0 * k_xy.mean())


def diversity_and_duplicate_rate(embeddings, threshold=0.98):
    if len(embeddings) < 2:
        return None, None
    sim = embeddings @ embeddings.T
    upper = sim[np.triu_indices(len(embeddings), k=1)]
    diversity = float(np.mean(1.0 - upper))
    duplicate_rate = float(np.mean(upper >= threshold))
    return diversity, duplicate_rate


def evaluate_metrics(all_generated, config, device):
    if config.no_clip:
        metrics = {"clip_model": config.clip_model, "clip_load_error": "disabled by --no-clip", "runs": {}}
        for run_name, rows in all_generated.items():
            metrics["runs"][run_name] = {
                "clip_text_image_mean": None,
                "clip_kid_proxy": None,
                "diversity_score": None,
                "duplicate_rate": None,
                "technical_image_score_mean": round(float(np.mean([x["technical_image_score"] for x in rows])), 4),
                "latency_seconds_mean": round(float(np.mean([x["latency_seconds"] for x in rows])), 4),
            }
        return metrics

    clip_model, clip_processor, error = try_load_clip(config.clip_model, device)
    metrics = {"clip_model": config.clip_model, "clip_load_error": error, "runs": {}}
    if clip_model is None:
        for run_name, rows in all_generated.items():
            metrics["runs"][run_name] = {
                "clip_text_image_mean": None,
                "clip_kid_proxy": None,
                "diversity_score": None,
                "duplicate_rate": None,
                "technical_image_score_mean": round(float(np.mean([x["technical_image_score"] for x in rows])), 4),
                "latency_seconds_mean": round(float(np.mean([x["latency_seconds"] for x in rows])), 4),
            }
        return metrics

    for run_name, rows in all_generated.items():
        clip_scores = clip_text_image_scores(clip_model, clip_processor, rows, "generated_image_path", device)
        gen_embeddings = clip_image_embeddings(
            clip_model,
            clip_processor,
            [row["generated_image_path"] for row in rows],
            device,
        )
        ref_embeddings = clip_image_embeddings(
            clip_model,
            clip_processor,
            [row["reference_image_path"] for row in rows],
            device,
        )
        kid_proxy = polynomial_mmd2(ref_embeddings, gen_embeddings)
        diversity, duplicate_rate = diversity_and_duplicate_rate(gen_embeddings)
        metrics["runs"][run_name] = {
            "clip_text_image_mean": round(float(np.mean(clip_scores)), 6),
            "clip_text_image_scores": [round(float(x), 6) for x in clip_scores],
            "clip_kid_proxy": None if kid_proxy is None else round(kid_proxy, 8),
            "diversity_score": None if diversity is None else round(diversity, 6),
            "duplicate_rate": None if duplicate_rate is None else round(duplicate_rate, 6),
            "technical_image_score_mean": round(float(np.mean([x["technical_image_score"] for x in rows])), 4),
            "latency_seconds_mean": round(float(np.mean([x["latency_seconds"] for x in rows])), 4),
        }
    del clip_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def compare_runs(metrics):
    runs = metrics.get("runs", {})
    a = runs.get("seed_only", {})
    b = runs.get("seed_plus_synthetic", {})
    comparison = {}
    for key, direction in [
        ("clip_text_image_mean", "higher_is_better"),
        ("clip_kid_proxy", "lower_is_better"),
        ("diversity_score", "higher_is_better_until_too_high"),
        ("duplicate_rate", "lower_is_better"),
        ("technical_image_score_mean", "higher_is_better"),
        ("latency_seconds_mean", "lower_is_better"),
    ]:
        av = a.get(key)
        bv = b.get(key)
        comparison[key] = {
            "seed_only": av,
            "seed_plus_synthetic": bv,
            "delta": None if av is None or bv is None else round(float(bv - av), 6),
            "direction": direction,
        }
    return comparison


def write_preview_grid(all_generated, output_path):
    rows_a = all_generated["seed_only"]
    rows_b = all_generated["seed_plus_synthetic"]
    thumb = 192
    label_h = 92
    margin = 16
    width = margin * 4 + thumb * 2 + 360
    height = margin + len(rows_a) * (thumb + label_h + margin)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 6), "Prompt", fill="#111827")
    draw.text((margin + 360 + margin, 6), "LoRA A: seed only", fill="#111827")
    draw.text((margin + 360 + margin * 2 + thumb, 6), "LoRA B: seed + synthetic", fill="#111827")
    y = margin + 18
    for row_a, row_b in zip(rows_a, rows_b):
        prompt = row_a["prompt"]
        wrapped = wrap_text(prompt, 42, 4)
        draw.text((margin, y + 12), wrapped, fill="#111827")
        img_a = Image.open(row_a["generated_image_path"]).convert("RGB").resize((thumb, thumb))
        img_b = Image.open(row_b["generated_image_path"]).convert("RGB").resize((thumb, thumb))
        x_a = margin + 360 + margin
        x_b = x_a + thumb + margin
        canvas.paste(img_a, (x_a, y))
        canvas.paste(img_b, (x_b, y))
        draw.text((x_a, y + thumb + 8), f"q={row_a['technical_image_score']} seed={row_a['seed']}", fill="#374151")
        draw.text((x_b, y + thumb + 8), f"q={row_b['technical_image_score']} seed={row_b['seed']}", fill="#374151")
        y += thumb + label_h + margin
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def write_metric_bars(comparison, output_path):
    keys = [
        "clip_text_image_mean",
        "clip_kid_proxy",
        "diversity_score",
        "technical_image_score_mean",
        "latency_seconds_mean",
    ]
    width = 1100
    row_h = 78
    margin = 24
    canvas = Image.new("RGB", (width, margin * 2 + row_h * len(keys)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 8), "LoRA A/B metric comparison", fill="#111827")
    y = margin + 16
    for key in keys:
        item = comparison.get(key, {})
        a = item.get("seed_only")
        b = item.get("seed_plus_synthetic")
        draw.text((margin, y), key, fill="#111827")
        if a is None or b is None:
            draw.text((margin + 320, y), "not available", fill="#6B7280")
            y += row_h
            continue
        max_value = max(abs(float(a)), abs(float(b)), 1e-8)
        bar_w_a = int(300 * abs(float(a)) / max_value)
        bar_w_b = int(300 * abs(float(b)) / max_value)
        x0 = margin + 320
        draw.rectangle((x0, y + 8, x0 + bar_w_a, y + 28), fill="#2563EB")
        draw.text((x0 + 310, y + 8), f"seed only: {a}", fill="#1F2937")
        draw.rectangle((x0, y + 38, x0 + bar_w_b, y + 58), fill="#059669")
        draw.text((x0 + 310, y + 38), f"seed+synthetic: {b}", fill="#1F2937")
        y += row_h
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def wrap_text(text, width, max_lines):
    words = text.split()
    lines = []
    current = []
    for word in words:
        if len(" ".join(current + [word])) <= width:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(" ".join(current))
    result = "\n".join(lines)
    if len(lines) == max_lines and len(words) > sum(len(line.split()) for line in lines):
        result += "..."
    return result


def write_report(config, data_info, metrics, comparison):
    out = Path(config.output_dir)
    report_path = out / "lora_ab_report.md"
    proxy_note = (
        "yes, local proxy split from synthetic_dataset because the repository has only Conceptual Captions text, not images"
        if data_info["proxy_mode"]
        else "no, used the provided seed image manifest"
    )
    lines = [
        "# LoRA A/B Experiment: seed-only vs seed+synthetic",
        "",
        "## Goal",
        "",
        "Check whether adding synthetic `text + image` pairs improves a diffusion model after LoRA fine-tuning.",
        "",
        "## Data",
        "",
        f"- Proxy seed mode: {proxy_note}",
        f"- Seed train size: {len(data_info['seed_train'])}",
        f"- Synthetic train size: {len(data_info['synthetic_train'])}",
        f"- Seed + synthetic train size: {len(data_info['seed_plus_synthetic_train'])}",
        f"- Validation size: {len(data_info['val'])}",
        f"- Test prompts/images: {len(data_info['test'])}",
        "",
        "## Training",
        "",
        f"- Base model: `{config.base_model}`",
        f"- LoRA rank: `{config.lora_rank}`",
        f"- Learning rate: `{config.learning_rate}`",
        f"- Train steps per run: `{config.train_steps}`",
        f"- Resolution: `{config.resolution}x{config.resolution}`",
        "",
        "## Metrics",
        "",
        "| Metric | Seed only | Seed + synthetic | Delta | Direction |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for key, item in comparison.items():
        lines.append(
            f"| `{key}` | {format_metric(item['seed_only'])} | {format_metric(item['seed_plus_synthetic'])} | "
            f"{format_metric(item['delta'])} | {item['direction']} |"
        )
    lines.extend(
        [
            "",
            f"- CLIP metric model: `{metrics.get('clip_model')}`",
            f"- CLIP load error: `{metrics.get('clip_load_error')}`",
            "",
            "## Artifacts",
            "",
            "- A/B preview grid: `eval/ab_preview_grid.png`",
            "- Metric bar chart: `eval/metric_comparison.png`",
            "- Full summary JSON: `summary.json`",
            "- LoRA adapters: `runs/*/adapters/`",
            "",
            "## Interpretation",
            "",
            "Use CLIP text-image similarity as the primary metric. Use KID/FID-like distribution metrics, diversity, duplicate rate, "
            "technical quality and latency as secondary checks. If seed+synthetic improves CLIPScore without a large loss of diversity "
            "or latency, the synthetic augmentation is useful for the chosen domain.",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def format_metric(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        return f"{value:.6g}"
    return str(value)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
    config = ExperimentConfig(**vars(args))
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", asdict(config))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_info = prepare_manifests(config)
    write_json(
        output_dir / "data_summary.json",
        {
            "proxy_mode": data_info["proxy_mode"],
            "seed_train_size": len(data_info["seed_train"]),
            "synthetic_train_size": len(data_info["synthetic_train"]),
            "seed_plus_synthetic_train_size": len(data_info["seed_plus_synthetic_train"]),
            "val_size": len(data_info["val"]),
            "test_size": len(data_info["test"]),
            "manifest_dir": data_info["manifest_dir"],
        },
    )

    train_summaries = {}
    if config.skip_training:
        print("Skipping training by request.", flush=True)
        existing_train_summary = output_dir / "train_summaries.json"
        if existing_train_summary.exists():
            train_summaries = json.loads(existing_train_summary.read_text(encoding="utf-8"))
    else:
        train_summaries["seed_only"] = train_lora(
            "seed_only",
            data_info["seed_train"],
            data_info["val"],
            config,
            device,
        )
        train_summaries["seed_plus_synthetic"] = train_lora(
            "seed_plus_synthetic",
            data_info["seed_plus_synthetic_train"],
            data_info["val"],
            config,
            device,
        )
    write_json(output_dir / "train_summaries.json", train_summaries)

    all_generated = {}
    for run_name in ["seed_only", "seed_plus_synthetic"]:
        adapter_dir = Path(config.output_dir) / "runs" / run_name / "adapters"
        if not adapter_dir.exists():
            raise RuntimeError(f"Adapter directory not found: {adapter_dir}")
        generated, latency_mean = generate_eval_images(run_name, adapter_dir, data_info["test"], config, device)
        all_generated[run_name] = generated
        print(f"{run_name} eval latency mean: {latency_mean}s", flush=True)

    metrics = evaluate_metrics(all_generated, config, device)
    comparison = compare_runs(metrics)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "comparison.json", comparison)
    write_preview_grid(all_generated, output_dir / "eval" / "ab_preview_grid.png")
    write_metric_bars(comparison, output_dir / "eval" / "metric_comparison.png")

    summary = {
        "config": asdict(config),
        "device": device,
        "data": {
            "proxy_mode": data_info["proxy_mode"],
            "seed_train_size": len(data_info["seed_train"]),
            "synthetic_train_size": len(data_info["synthetic_train"]),
            "seed_plus_synthetic_train_size": len(data_info["seed_plus_synthetic_train"]),
            "val_size": len(data_info["val"]),
            "test_size": len(data_info["test"]),
        },
        "train_summaries": train_summaries,
        "metrics": metrics,
        "comparison": comparison,
    }
    write_json(output_dir / "summary.json", summary)
    report_path = write_report(config, data_info, metrics, comparison)
    print(json.dumps({"summary": str(output_dir / "summary.json"), "report": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
