import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "images"
PLOTS_DIR = ROOT / "plots"
MANIFEST_PATH = ROOT / "synthetic_manifest.csv"
STATS_PATH = ROOT / "synthetic_stats.json"
PREVIEW_PATH = ROOT / "synthetic_dataset_preview.png"
SEED = 42


PROMPTS = [
    ("sports", "football player in red uniform running across a wet stadium field"),
    ("sports", "basketball player jumping near the hoop under bright arena lights"),
    ("sports", "tennis racket and ball on a blue court with strong shadows"),
    ("sports", "cyclist on a mountain road during a cloudy morning race"),
    ("animals_nature", "golden retriever sitting near a small river in autumn forest"),
    ("animals_nature", "white sheep standing on a snowy hill under pale sky"),
    ("animals_nature", "green turtle on a clean white studio background"),
    ("animals_nature", "deer silhouette near trees with orange sunset behind"),
    ("people_entertainment", "singer holding a microphone on a small concert stage"),
    ("people_entertainment", "actor walking on a red carpet near photographers"),
    ("people_entertainment", "business woman presenting a chart in a modern office"),
    ("people_entertainment", "person dancing in colorful clothes during a city festival"),
    ("places_objects", "minimal living room with green couch and wooden coffee table"),
    ("places_objects", "safe deposit box with paper money on a white background"),
    ("places_objects", "modern bridge over a river with glass buildings nearby"),
    ("places_objects", "bright bus station with empty benches and morning light"),
]


PALETTE = {
    "sports": ((226, 88, 55), (31, 78, 121)),
    "animals_nature": ((67, 139, 84), (225, 177, 91)),
    "people_entertainment": ((190, 84, 137), (79, 61, 127)),
    "places_objects": ((56, 117, 165), (218, 196, 145)),
}


def font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


TITLE_FONT = font(24)
TEXT_FONT = font(18)
SMALL_FONT = font(14)


def wrap(text, max_len=34):
    words = text.split()
    lines = []
    current = []
    for word in words:
        candidate = " ".join(current + [word])
        if len(candidate) > max_len and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def draw_person(draw, x, y, shirt=(226, 88, 55)):
    draw.ellipse((x - 22, y - 90, x + 22, y - 46), fill=(245, 205, 170), outline=(33, 41, 54), width=3)
    draw.rounded_rectangle((x - 34, y - 45, x + 34, y + 55), radius=18, fill=shirt, outline=(33, 41, 54), width=3)
    draw.line((x - 34, y - 10, x - 90, y + 35), fill=(33, 41, 54), width=8)
    draw.line((x + 34, y - 10, x + 90, y + 35), fill=(33, 41, 54), width=8)
    draw.line((x - 16, y + 55, x - 48, y + 125), fill=(33, 41, 54), width=9)
    draw.line((x + 16, y + 55, x + 48, y + 125), fill=(33, 41, 54), width=9)


def draw_sports_scene(draw, prompt):
    draw.rectangle((0, 0, 768, 352), fill=(56, 117, 165))
    draw.rectangle((0, 255, 768, 352), fill=(48, 133, 76))
    for x in range(0, 768, 96):
        draw.line((x, 255, x + 60, 352), fill=(235, 246, 235), width=3)
    draw_person(draw, 365, 230, shirt=(226, 88, 55))
    if "basketball" in prompt:
        draw.ellipse((515, 78, 600, 163), fill=(218, 117, 54), outline=(33, 41, 54), width=4)
        draw.rectangle((604, 82, 690, 100), fill=(230, 230, 230), outline=(33, 41, 54), width=3)
        draw.ellipse((635, 100, 700, 165), outline=(255, 255, 255), width=6)
    elif "tennis" in prompt:
        draw.ellipse((500, 95, 610, 165), outline=(33, 41, 54), width=8)
        draw.line((575, 155, 655, 235), fill=(33, 41, 54), width=8)
        draw.ellipse((205, 92, 245, 132), fill=(230, 240, 85), outline=(33, 41, 54), width=3)
    elif "cyclist" in prompt:
        draw.ellipse((225, 245, 325, 345), outline=(33, 41, 54), width=9)
        draw.ellipse((455, 245, 555, 345), outline=(33, 41, 54), width=9)
        draw.line((275, 295, 405, 190), fill=(33, 41, 54), width=8)
        draw.line((405, 190, 505, 295), fill=(33, 41, 54), width=8)
        draw.line((275, 295, 505, 295), fill=(33, 41, 54), width=8)
    else:
        draw.ellipse((520, 230, 600, 310), fill=(255, 255, 255), outline=(33, 41, 54), width=4)
        draw.arc((520, 230, 600, 310), 90, 270, fill=(33, 41, 54), width=3)


def draw_nature_scene(draw, prompt):
    draw.rectangle((0, 0, 768, 250), fill=(166, 210, 232))
    draw.rectangle((0, 250, 768, 352), fill=(92, 148, 83))
    for x in [85, 175, 610, 700]:
        draw.rectangle((x, 150, x + 24, 285), fill=(106, 74, 45))
        draw.polygon([(x + 12, 55), (x - 60, 180), (x + 84, 180)], fill=(48, 113, 70))
    if "retriever" in prompt or "dog" in prompt:
        draw.ellipse((270, 210, 470, 310), fill=(207, 150, 72), outline=(33, 41, 54), width=4)
        draw.ellipse((420, 170, 520, 260), fill=(207, 150, 72), outline=(33, 41, 54), width=4)
        draw.ellipse((455, 200, 470, 215), fill=(33, 41, 54))
        draw.line((520, 235, 580, 205), fill=(207, 150, 72), width=14)
    elif "sheep" in prompt:
        for x in [250, 300, 350, 400]:
            draw.ellipse((x, 190, x + 105, 285), fill=(245, 245, 238), outline=(33, 41, 54), width=3)
        draw.ellipse((430, 215, 505, 280), fill=(60, 60, 60), outline=(33, 41, 54), width=3)
        draw.rectangle((0, 285, 768, 352), fill=(240, 245, 247))
    elif "turtle" in prompt:
        draw.ellipse((260, 180, 510, 305), fill=(72, 138, 78), outline=(33, 41, 54), width=5)
        draw.ellipse((495, 210, 585, 270), fill=(80, 150, 86), outline=(33, 41, 54), width=4)
        draw.ellipse((530, 232, 542, 244), fill=(33, 41, 54))
    else:
        draw.ellipse((320, 190, 500, 295), fill=(150, 91, 45), outline=(33, 41, 54), width=4)
        draw.ellipse((460, 145, 550, 225), fill=(150, 91, 45), outline=(33, 41, 54), width=4)
        draw.line((500, 150, 470, 90), fill=(33, 41, 54), width=5)
        draw.line((500, 150, 540, 90), fill=(33, 41, 54), width=5)
        draw.ellipse((500, 175, 512, 187), fill=(33, 41, 54))


def draw_people_scene(draw, prompt):
    draw.rectangle((0, 0, 768, 352), fill=(72, 54, 126))
    for x in range(80, 720, 130):
        draw.polygon([(x, 0), (x + 55, 0), (x + 8, 352), (x - 48, 352)], fill=(255, 224, 123))
    draw.rectangle((0, 275, 768, 352), fill=(55, 43, 82))
    if "singer" in prompt or "stage" in prompt:
        draw_person(draw, 360, 215, shirt=(196, 67, 132))
        draw.line((455, 145, 455, 285), fill=(33, 41, 54), width=8)
        draw.ellipse((430, 120, 480, 158), fill=(33, 41, 54))
    elif "red carpet" in prompt:
        draw.polygon([(250, 352), (520, 352), (460, 190), (310, 190)], fill=(190, 40, 55))
        draw_person(draw, 385, 220, shirt=(30, 30, 40))
        for x in [110, 610]:
            draw.rectangle((x, 150, x + 85, 210), fill=(240, 240, 240), outline=(33, 41, 54), width=3)
            draw.ellipse((x + 25, 165, x + 55, 195), fill=(33, 41, 54))
    elif "business" in prompt or "office" in prompt:
        draw.rectangle((460, 95, 690, 250), fill=(245, 245, 245), outline=(33, 41, 54), width=4)
        draw.line((500, 225, 555, 170), fill=(47, 125, 77), width=8)
        draw.line((555, 170, 630, 135), fill=(226, 88, 55), width=8)
        draw_person(draw, 260, 230, shirt=(56, 117, 165))
    else:
        draw_person(draw, 300, 230, shirt=(226, 88, 55))
        draw_person(draw, 470, 230, shirt=(67, 139, 84))
        draw.arc((250, 115, 520, 330), 20, 160, fill=(255, 255, 255), width=6)


def draw_places_scene(draw, prompt):
    draw.rectangle((0, 0, 768, 352), fill=(224, 236, 240))
    draw.rectangle((0, 270, 768, 352), fill=(210, 197, 174))
    if "living room" in prompt:
        draw.rectangle((85, 70, 685, 285), fill=(245, 240, 230), outline=(33, 41, 54), width=4)
        draw.rounded_rectangle((220, 190, 550, 280), radius=24, fill=(72, 138, 95), outline=(33, 41, 54), width=4)
        draw.rectangle((330, 285, 450, 325), fill=(130, 86, 50), outline=(33, 41, 54), width=3)
        draw.ellipse((120, 95, 190, 165), fill=(255, 221, 120), outline=(33, 41, 54), width=3)
    elif "safe deposit" in prompt:
        draw.rounded_rectangle((240, 100, 530, 285), radius=18, fill=(160, 170, 178), outline=(33, 41, 54), width=6)
        draw.ellipse((350, 160, 425, 235), fill=(220, 224, 228), outline=(33, 41, 54), width=5)
        for x in [130, 575]:
            draw.rectangle((x, 210, x + 115, 260), fill=(103, 156, 86), outline=(33, 41, 54), width=3)
            draw.text((x + 35, 224), "$", fill=(255, 255, 255), font=TITLE_FONT)
    elif "bridge" in prompt:
        draw.rectangle((0, 230, 768, 352), fill=(95, 166, 205))
        draw.arc((120, 130, 650, 360), 190, 350, fill=(33, 41, 54), width=12)
        for x in range(170, 650, 90):
            draw.line((x, 190, x, 300), fill=(33, 41, 54), width=5)
        draw.rectangle((80, 195, 700, 225), fill=(226, 226, 220), outline=(33, 41, 54), width=4)
    else:
        draw.rectangle((85, 80, 690, 260), fill=(235, 235, 230), outline=(33, 41, 54), width=5)
        for x in range(140, 640, 120):
            draw.rectangle((x, 125, x + 80, 215), fill=(178, 210, 230), outline=(33, 41, 54), width=3)
        draw.rounded_rectangle((220, 255, 560, 320), radius=14, fill=(95, 95, 95), outline=(33, 41, 54), width=3)


def synthetic_clip_score(prompt, domain):
    tokens = set(prompt.split())
    domain_tokens = {
        "sports": {"player", "field", "court", "race", "ball"},
        "animals_nature": {"river", "forest", "sheep", "turtle", "deer", "trees"},
        "people_entertainment": {"singer", "actor", "person", "stage", "office"},
        "places_objects": {"room", "box", "bridge", "station", "table"},
    }
    overlap = len(tokens & domain_tokens[domain])
    return round(0.71 + min(overlap, 4) * 0.045 + random.random() * 0.035, 3)


def generate_image(idx, domain, prompt):
    width, height = 768, 512
    bg = (248, 244, 236)
    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)
    if domain == "sports":
        draw_sports_scene(draw, prompt)
    elif domain == "animals_nature":
        draw_nature_scene(draw, prompt)
    elif domain == "people_entertainment":
        draw_people_scene(draw, prompt)
    else:
        draw_places_scene(draw, prompt)
    draw.rounded_rectangle((28, 28, width - 28, height - 28), radius=28, outline=(33, 41, 54), width=4)
    draw.rectangle((0, height - 160, width, height), fill=(250, 248, 242))
    draw.text((42, height - 140), domain, fill=(33, 41, 54), font=TITLE_FONT)
    draw.multiline_text((42, height - 104), wrap(prompt, 64), fill=(65, 70, 80), font=TEXT_FONT, spacing=5)
    draw.text((width - 170, 42), f"synthetic #{idx:02d}", fill=(33, 41, 54), font=SMALL_FONT)
    return image


def make_bar_chart(counts):
    width, height = 900, 520
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((50, 30), "Synthetic demo dataset: domain distribution", fill="black", font=TITLE_FONT)
    labels = list(PALETTE.keys())
    max_value = max(counts.values())
    x0, base_y = 80, 430
    bar_w = 150
    gap = 45
    for idx, label in enumerate(labels):
        value = counts[label]
        h = int((value / max_value) * 280)
        x = x0 + idx * (bar_w + gap)
        color = PALETTE[label][0]
        draw.rectangle((x, base_y - h, x + bar_w, base_y), fill=color)
        draw.text((x + 55, base_y - h - 30), str(value), fill="black", font=TEXT_FONT)
        draw.text((x, base_y + 15), label.replace("_", "\n"), fill="black", font=SMALL_FONT)
    path = PLOTS_DIR / "synthetic_domain_distribution.png"
    image.save(path)


def make_preview(records):
    thumb_w, thumb_h = 320, 220
    cols = 4
    rows = math.ceil(min(len(records), 8) / cols)
    image = Image.new("RGB", (cols * thumb_w, rows * thumb_h + 70), "white")
    draw = ImageDraw.Draw(image)
    draw.text((28, 20), "Synthetic text-image demo samples", fill="black", font=TITLE_FONT)
    for idx, rec in enumerate(records[:8]):
        img = Image.open(ROOT / rec["image_path"]).resize((thumb_w - 24, thumb_h - 58))
        x = (idx % cols) * thumb_w + 12
        y = (idx // cols) * thumb_h + 64
        image.paste(img, (x, y))
        draw.text((x, y + thumb_h - 48), rec["domain_tag"], fill="black", font=SMALL_FONT)
        draw.text((x, y + thumb_h - 28), f"quality={rec['quality_score']}", fill="black", font=SMALL_FONT)
    image.save(PREVIEW_PATH)


def main():
    random.seed(SEED)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for idx, (domain, prompt) in enumerate(PROMPTS, start=1):
        image_name = f"synthetic_{idx:03d}_{domain}.png"
        image_path = IMAGES_DIR / image_name
        image = generate_image(idx, domain, prompt)
        image.save(image_path)
        score = synthetic_clip_score(prompt, domain)
        records.append(
            {
                "id": f"synthetic_{idx:03d}",
                "prompt": prompt,
                "image_path": f"images/{image_name}",
                "domain_tag": domain,
                "source_type": "synthetic_demo",
                "generator": "programmatic_placeholder_generator",
                "base_model_checkpoint": "demo-no-diffusion-checkpoint",
                "seed": SEED + idx,
                "resolution": "768x512",
                "quality_score": score,
                "quality_flag": "accepted" if score >= 0.75 else "review",
            }
        )

    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    counts = Counter(rec["domain_tag"] for rec in records)
    stats = {
        "num_samples": len(records),
        "domain_distribution": counts,
        "avg_quality_score": round(sum(rec["quality_score"] for rec in records) / len(records), 3),
        "accepted_samples": sum(1 for rec in records if rec["quality_flag"] == "accepted"),
        "generator_note": "Demo artifact. Images are schematic programmatic placeholders, not diffusion outputs.",
    }
    STATS_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    make_bar_chart(counts)
    make_preview(records)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
