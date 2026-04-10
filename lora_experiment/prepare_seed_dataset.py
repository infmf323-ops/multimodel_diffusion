import argparse
import csv
import json
import os
from pathlib import Path

from datasets import load_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ID = "pasindu/google_conceptual_captions_20000"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "lora_experiment" / "seed_dataset" / "google_conceptual_captions_20000"
DEFAULT_HF_HOME = REPO_ROOT / "hf_cache"


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a local image-caption seed manifest.")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset_id, split=args.split)
    rows = []

    for index, example in enumerate(dataset):
        if len(rows) >= args.limit:
            break
        caption = str(example.get("caption", "")).strip()
        image = example.get("image_data")
        if not caption or image is None:
            continue

        image = image.convert("RGB")
        sample_id = f"cc_seed_{len(rows) + 1:04d}"
        image_name = f"{sample_id}.jpg"
        image_path = images_dir / image_name
        image.save(image_path, format="JPEG", quality=92)

        rows.append(
            {
                "id": sample_id,
                "prompt": caption,
                "image_path": f"images/{image_name}",
                "source_dataset": args.dataset_id,
                "source_split": args.split,
                "source_index": index,
                "width": image.width,
                "height": image.height,
            }
        )

    manifest_path = output_dir / "seed_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "prompt",
                "image_path",
                "source_dataset",
                "source_split",
                "source_index",
                "width",
                "height",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "dataset_id": args.dataset_id,
        "split": args.split,
        "num_samples": len(rows),
        "manifest_path": str(manifest_path),
        "images_dir": str(images_dir),
        "note": "Local seed image-caption subset for LoRA A/B experiment.",
    }
    (output_dir / "seed_dataset_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
