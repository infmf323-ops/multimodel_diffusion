import csv
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont, ImageStat


ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "images"
PLOTS_DIR = ROOT / "plots"
MANIFEST_PATH = ROOT / "synthetic_manifest.csv"
STATS_PATH = ROOT / "synthetic_stats.json"
PREVIEW_PATH = ROOT / "synthetic_dataset_preview.png"

MODEL_ID = os.getenv("DIFFUSION_MODEL_ID", "segmind/tiny-sd")
BASE_SEED = int(os.getenv("DIFFUSION_BASE_SEED", "20260410"))
WIDTH = int(os.getenv("DIFFUSION_WIDTH", "512"))
HEIGHT = int(os.getenv("DIFFUSION_HEIGHT", "512"))
STEPS = int(os.getenv("DIFFUSION_STEPS", "25"))
GUIDANCE_SCALE = float(os.getenv("DIFFUSION_GUIDANCE_SCALE", "7.0"))
LIMIT = int(os.getenv("DIFFUSION_LIMIT", "0"))

NEGATIVE_PROMPT = (
    "low quality, blurry, distorted anatomy, extra limbs, text, watermark, logo, "
    "cropped subject, jpeg artifacts"
)

PROMPTS = [
    ("sports", "football player in red uniform running across a wet stadium field"),
    ("sports", "basketball player jumping near the hoop under bright arena lights"),
    ("sports", "tennis racket and ball on a blue court with strong shadows"),
    ("sports", "cyclist on a mountain road during a cloudy morning race"),
    ("sports", "goalkeeper catching a ball during a night soccer match"),
    ("sports", "runner crossing a finish line in a city marathon"),
    ("sports", "swimmer diving into a blue Olympic pool"),
    ("sports", "ski jumper flying over a snowy mountain slope"),
    ("sports", "boxer training with a punching bag in a small gym"),
    ("sports", "baseball player swinging a bat in a sunny stadium"),
    ("sports", "ice hockey player skating fast with the puck"),
    ("sports", "surfer riding a large wave at sunset"),
    ("sports", "gymnast balancing on a beam under arena lights"),
    ("animals_nature", "golden retriever sitting near a small river in autumn forest"),
    ("animals_nature", "white sheep standing on a snowy hill under pale sky"),
    ("animals_nature", "green turtle on a clean white studio background"),
    ("animals_nature", "deer silhouette near trees with orange sunset behind"),
    ("animals_nature", "brown horse running through a misty meadow"),
    ("animals_nature", "red fox standing beside a fallen tree in a forest"),
    ("animals_nature", "small bird perched on a branch after rain"),
    ("animals_nature", "gray cat sleeping near a window with plants"),
    ("animals_nature", "elephant walking near a watering hole in warm light"),
    ("animals_nature", "penguins standing together on an icy shore"),
    ("animals_nature", "butterfly on a purple flower in a sunny garden"),
    ("animals_nature", "owl sitting on a wooden fence at dusk"),
    ("animals_nature", "dolphin jumping above calm ocean water"),
    ("people_entertainment", "singer holding a microphone on a small concert stage"),
    ("people_entertainment", "actor walking on a red carpet near photographers"),
    ("people_entertainment", "business woman presenting a chart in a modern office"),
    ("people_entertainment", "person dancing in colorful clothes during a city festival"),
    ("people_entertainment", "chef preparing pasta in a bright restaurant kitchen"),
    ("people_entertainment", "painter working on a large canvas in a studio"),
    ("people_entertainment", "teacher explaining a diagram on a classroom board"),
    ("people_entertainment", "musician playing an acoustic guitar near a cafe window"),
    ("people_entertainment", "photographer taking a portrait in a small studio"),
    ("people_entertainment", "two friends laughing at an outdoor street market"),
    ("people_entertainment", "child building a colorful toy tower on a carpet"),
    ("people_entertainment", "dancer practicing ballet in a mirrored rehearsal room"),
    ("places_objects", "minimal living room with green couch and wooden coffee table"),
    ("places_objects", "safe deposit box with paper money on a white background"),
    ("places_objects", "modern bridge over a river with glass buildings nearby"),
    ("places_objects", "bright bus station with empty benches and morning light"),
    ("places_objects", "small bookstore aisle with warm lamps and wooden shelves"),
    ("places_objects", "laptop and notebook on a clean office desk"),
    ("places_objects", "old bicycle leaning against a brick wall"),
    ("places_objects", "glass teapot and cups on a kitchen table"),
    ("places_objects", "quiet train platform with a yellow safety line"),
    ("places_objects", "cozy bedroom with blue blanket and bedside lamp"),
    ("places_objects", "city street corner with traffic lights after rain"),
    ("places_objects", "wooden fishing boat tied near a calm lake dock"),
]


def font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


TITLE_FONT = font(20)
TEXT_FONT = font(14)


def enhanced_prompt(prompt):
    return (
        f"{prompt}, realistic photo, high quality, detailed, natural lighting, "
        "sharp focus, dataset sample"
    )


def technical_quality_score(image):
    rgb = image.convert("RGB")
    gray = image.convert("L")
    stat = ImageStat.Stat(rgb)
    gray_stat = ImageStat.Stat(gray)
    contrast = min(gray_stat.stddev[0] / 64.0, 1.0)

    arr = np.asarray(rgb).astype(np.float32) / 255.0
    channel_max = arr.max(axis=2)
    channel_min = arr.min(axis=2)
    saturation = np.mean((channel_max - channel_min) / np.maximum(channel_max, 1e-6))
    brightness = np.mean(channel_max)
    brightness_penalty = abs(brightness - 0.55) / 0.55

    color_variation = min(float(np.mean(stat.stddev)) / 72.0, 1.0)
    score = 0.45 * contrast + 0.35 * float(saturation) + 0.20 * color_variation
    score = max(0.0, min(1.0, score * (1.0 - 0.35 * brightness_penalty)))
    return round(float(score), 3)


def load_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
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


def write_manifest(rows):
    fieldnames = [
        "id",
        "prompt",
        "generation_prompt",
        "negative_prompt",
        "image_path",
        "domain_tag",
        "source_type",
        "generator",
        "base_model_checkpoint",
        "seed",
        "resolution",
        "num_inference_steps",
        "guidance_scale",
        "quality_score",
        "quality_score_type",
        "quality_flag",
    ]
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_domain_plot(rows):
    counts = Counter(row["domain_tag"] for row in rows)
    width, height = 980, 520
    image = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(image)
    draw.text((40, 30), "Domain distribution in diffusion synthetic dataset", fill="#111827", font=TITLE_FONT)

    domains = list(counts.keys())
    max_count = max(counts.values()) if counts else 1
    bar_width = 150
    gap = 60
    base_y = 420
    colors = ["#2563eb", "#16a34a", "#c026d3", "#f97316"]
    for idx, domain in enumerate(domains):
        x = 70 + idx * (bar_width + gap)
        bar_height = int((counts[domain] / max_count) * 280)
        draw.rectangle((x, base_y - bar_height, x + bar_width, base_y), fill=colors[idx % len(colors)])
        draw.text((x + 55, base_y - bar_height - 28), str(counts[domain]), fill="#111827", font=TEXT_FONT)
        draw.text((x, base_y + 18), domain, fill="#374151", font=TEXT_FONT)
    draw.line((50, base_y, width - 60, base_y), fill="#111827", width=2)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    image.save(PLOTS_DIR / "synthetic_domain_distribution.png")


def build_preview(rows):
    thumbs = []
    for row in rows:
        path = ROOT / row["image_path"]
        thumb = Image.open(path).convert("RGB").resize((220, 220))
        card = Image.new("RGB", (260, 320), "#ffffff")
        draw = ImageDraw.Draw(card)
        card.paste(thumb, (20, 18))
        draw.text((20, 252), row["domain_tag"], fill="#111827", font=TEXT_FONT)
        draw.text((20, 278), f"q={row['quality_score']}", fill="#374151", font=TEXT_FONT)
        thumbs.append(card)

    cols = 4
    rows_count = (len(thumbs) + cols - 1) // cols
    preview = Image.new("RGB", (cols * 260, rows_count * 320), "#e5e7eb")
    for idx, thumb in enumerate(thumbs):
        x = (idx % cols) * 260
        y = (idx // cols) * 320
        preview.paste(thumb, (x, y))
    preview.save(PREVIEW_PATH)


def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for stale_image in IMAGES_DIR.glob("synthetic_*.png"):
        stale_image.unlink()

    selected_prompts = PROMPTS[:LIMIT] if LIMIT > 0 else PROMPTS
    pipe, device = load_pipeline()
    rows = []

    for idx, (domain, prompt) in enumerate(selected_prompts, start=1):
        seed = BASE_SEED + idx
        generation_prompt = enhanced_prompt(prompt)
        generator = torch.Generator(device=device).manual_seed(seed)
        result = pipe(
            generation_prompt,
            negative_prompt=NEGATIVE_PROMPT,
            width=WIDTH,
            height=HEIGHT,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
        )
        image = result.images[0]
        image_name = f"synthetic_{idx:03d}_{domain}.png"
        image.save(IMAGES_DIR / image_name)
        score = technical_quality_score(image)
        rows.append(
            {
                "id": f"synthetic_{idx:03d}",
                "prompt": prompt,
                "generation_prompt": generation_prompt,
                "negative_prompt": NEGATIVE_PROMPT,
                "image_path": f"images/{image_name}",
                "domain_tag": domain,
                "source_type": "synthetic_diffusion",
                "generator": "diffusers.StableDiffusionPipeline",
                "base_model_checkpoint": MODEL_ID,
                "seed": seed,
                "resolution": f"{WIDTH}x{HEIGHT}",
                "num_inference_steps": STEPS,
                "guidance_scale": GUIDANCE_SCALE,
                "quality_score": score,
                "quality_score_type": "technical_image_score",
                "quality_flag": "accepted" if score >= 0.25 else "review",
            }
        )
        print(f"generated {rows[-1]['id']} -> {rows[-1]['image_path']} ({domain}, q={score})")

    write_manifest(rows)
    build_domain_plot(rows)
    build_preview(rows)

    stats = {
        "num_samples": len(rows),
        "domain_distribution": dict(Counter(row["domain_tag"] for row in rows)),
        "avg_quality_score": round(float(np.mean([row["quality_score"] for row in rows])), 3),
        "accepted_samples": sum(row["quality_flag"] == "accepted" for row in rows),
        "source_type": "synthetic_diffusion",
        "generator": "diffusers.StableDiffusionPipeline",
        "base_model_checkpoint": MODEL_ID,
        "resolution": f"{WIDTH}x{HEIGHT}",
        "num_inference_steps": STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "base_seed": BASE_SEED,
        "device": device,
        "generator_note": "Real diffusion model outputs generated from text prompts.",
    }
    STATS_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
