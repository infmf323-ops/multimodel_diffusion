import csv
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "conceptual_captions_sample_100.tsv"
ARTIFACTS_DIR = ROOT / "artifacts"
EXPERIMENTS_DIR = ARTIFACTS_DIR / "experiments"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
MODELS_DIR = ARTIFACTS_DIR / "models"


SPORTS_KEYWORDS = {
    "football",
    "soccer",
    "sports",
    "playoff",
    "game",
    "player",
    "tournament",
    "birdie",
    "golf",
    "racecar",
    "skates",
    "camp",
    "scoring",
}
PEOPLE_KEYWORDS = {
    "actor",
    "actress",
    "artist",
    "musician",
    "celebrities",
    "celebrity",
    "woman",
    "man",
    "girl",
    "businessman",
    "students",
    "people",
    "person",
    "daughters",
    "bride",
    "groom",
    "fiancee",
    "premiere",
    "stage",
    "singer",
    "star",
}
NATURE_KEYWORDS = {
    "animal",
    "deer",
    "giraffe",
    "dog",
    "turtle",
    "zebra",
    "sheep",
    "lion",
    "puppy",
    "retriever",
    "river",
    "woodland",
    "snow",
    "mountain",
    "field",
    "tree",
    "shrubs",
    "plants",
    "ecosystems",
    "lake",
    "water",
    "forest",
    "forests",
    "wildfire",
    "lawn",
    "island",
    "boat",
}

STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "and",
    "in",
    "on",
    "at",
    "to",
    "for",
    "with",
    "during",
    "their",
    "this",
    "that",
    "is",
    "are",
    "was",
    "were",
    "it",
    "its",
    "my",
    "your",
}

LABEL_ORDER = ["sports", "people_entertainment", "animals_nature", "places_objects"]


def ensure_dirs():
    for path in [ARTIFACTS_DIR, EXPERIMENTS_DIR, PLOTS_DIR, MODELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("sports team", "sports_team")
    text = text.replace("'", "")
    text = re.sub(r"[^a-z0-9_!?.,\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_words(text: str):
    return re.findall(r"[a-z_]+", normalize_text(text))


def model_text(text: str) -> str:
    text = normalize_text(text)
    replacements = {
        "sports_team": "team_token",
        "actor": "person_token",
        "actress": "person_token",
        "artist": "person_token",
        "musician": "person_token",
        "woman": "person_token",
        "man": "person_token",
        "girl": "person_token",
        "people": "person_token",
        "person": "person_token",
        "businessman": "person_token",
        "students": "person_token",
        "fiancee": "person_token",
        "daughters": "person_token",
        "dog": "animal_token",
        "deer": "animal_token",
        "giraffe": "animal_token",
        "turtle": "animal_token",
        "zebra": "animal_token",
        "sheep": "animal_token",
        "retriever": "animal_token",
        "puppy": "animal_token",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def assign_label(text: str) -> str:
    tokens = set(tokenize_words(text))
    if tokens & SPORTS_KEYWORDS:
        return "sports"
    if tokens & PEOPLE_KEYWORDS:
        return "people_entertainment"
    if tokens & NATURE_KEYWORDS:
        return "animals_nature"
    return "places_objects"


def make_group_key(text: str) -> str:
    tokens = []
    for token in tokenize_words(text):
        if token in SPORTS_KEYWORDS:
            token = "sports"
        elif token in PEOPLE_KEYWORDS:
            token = "people"
        elif token in NATURE_KEYWORDS:
            token = "nature"
        if token not in STOPWORDS:
            tokens.append(token)
    return " ".join(tokens[:4]) or "empty"


def load_rows():
    rows = []
    with DATA_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for raw in reader:
            caption = normalize_text(raw["caption"])
            label = assign_label(caption)
            group_key = make_group_key(caption)
            rows.append(
                {
                    "id": int(raw["id"]),
                    "caption": caption,
                    "label": label,
                    "group_key": group_key,
                }
            )
    deduped = []
    seen = set()
    for row in rows:
        if row["caption"] in seen:
            continue
        seen.add(row["caption"])
        deduped.append(row)
    return assign_group_splits(deduped)


def assign_group_splits(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_key"]].append(row)

    groups_by_label = defaultdict(list)
    for group_key, items in groups.items():
        label = Counter(item["label"] for item in items).most_common(1)[0][0]
        groups_by_label[label].append(group_key)

    assignment = {}
    for label, group_keys in groups_by_label.items():
        ordered = sorted(group_keys, key=lambda key: hashlib.md5(key.encode("utf-8")).hexdigest())
        count = len(ordered)
        if count == 1:
            split_sizes = {"train": 1, "val": 0, "test": 0}
        elif count == 2:
            split_sizes = {"train": 1, "val": 0, "test": 1}
        else:
            n_val = max(1, round(count * 0.15))
            n_test = max(1, round(count * 0.15))
            n_train = count - n_val - n_test
            if n_train < 1:
                n_train = 1
                if n_val > n_test:
                    n_val -= 1
                else:
                    n_test -= 1
            split_sizes = {"train": n_train, "val": n_val, "test": n_test}

        cursor = 0
        for split_name in ["train", "val", "test"]:
            for group_key in ordered[cursor : cursor + split_sizes[split_name]]:
                assignment[group_key] = split_name
            cursor += split_sizes[split_name]

    result = []
    for row in rows:
        row = dict(row)
        row["split"] = assignment[row["group_key"]]
        result.append(row)
    return result


class TfidfVectorizer:
    def __init__(self, analyzer="word", ngram_range=(1, 1), max_features=150):
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.vocab = {}
        self.idf = None

    def _doc_features(self, text: str):
        if self.analyzer == "word":
            tokens = re.findall(r"[a-z_]+", model_text(text))
            ngrams = []
            for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
                for idx in range(len(tokens) - n + 1):
                    ngrams.append(" ".join(tokens[idx : idx + n]))
            return ngrams
        padded = f" {model_text(text)} "
        ngrams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for idx in range(len(padded) - n + 1):
                ngrams.append(padded[idx : idx + n])
        return ngrams

    def fit(self, texts):
        doc_freq = Counter()
        for text in texts:
            doc_freq.update(set(self._doc_features(text)))
        ranked = sorted(doc_freq.items(), key=lambda item: (-item[1], item[0]))[: self.max_features]
        self.vocab = {token: idx for idx, (token, _) in enumerate(ranked)}
        n_docs = len(texts)
        self.idf = np.ones(len(self.vocab), dtype=np.float64)
        for token, idx in self.vocab.items():
            self.idf[idx] = math.log((1 + n_docs) / (1 + doc_freq[token])) + 1.0
        return self

    def transform(self, texts):
        matrix = np.zeros((len(texts), len(self.vocab)), dtype=np.float64)
        for row_idx, text in enumerate(texts):
            counts = Counter(self._doc_features(text))
            total = sum(counts.values()) or 1
            for token, count in counts.items():
                if token not in self.vocab:
                    continue
                col_idx = self.vocab[token]
                tf = count / total
                matrix[row_idx, col_idx] = tf * self.idf[col_idx]
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


def extra_features(texts):
    feats = []
    for text in texts:
        tokens = re.findall(r"[a-z_]+", model_text(text))
        feats.append(
            [
                len(tokens),
                len(text),
                text.count("!"),
                int("sports_team" in text),
                int("person" in tokens),
                int(any(token in SPORTS_KEYWORDS for token in tokens)),
                int(any(token in PEOPLE_KEYWORDS for token in tokens)),
                int(any(token in NATURE_KEYWORDS for token in tokens)),
            ]
        )
    arr = np.asarray(feats, dtype=np.float64)
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (arr - mean) / std


@dataclass
class FeatureBundle:
    vectorizer: TfidfVectorizer
    use_extra_features: bool

    def fit_transform(self, texts):
        base = self.vectorizer.fit(texts).transform(texts)
        if not self.use_extra_features:
            return base
        return np.hstack([base, extra_features(texts)])

    def transform(self, texts):
        base = self.vectorizer.transform(texts)
        if not self.use_extra_features:
            return base
        return np.hstack([base, extra_features(texts)])


class SoftmaxRegression:
    def __init__(self, learning_rate=0.3, epochs=700, l2=1e-3, class_weight=None):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.class_weight = class_weight
        self.weights = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = int(y.max()) + 1
        Xb = np.hstack([X, np.ones((n_samples, 1), dtype=np.float64)])
        self.weights = np.zeros((n_features + 1, n_classes), dtype=np.float64)
        y_one_hot = np.eye(n_classes)[y]
        sample_weights = np.ones(n_samples, dtype=np.float64)
        if self.class_weight == "balanced":
            class_counts = np.bincount(y, minlength=n_classes)
            for idx, count in enumerate(class_counts):
                if count > 0:
                    sample_weights[y == idx] = n_samples / (n_classes * count)
        for _ in range(self.epochs):
            logits = Xb @ self.weights
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            diff = (probs - y_one_hot) * sample_weights[:, None]
            grad = (Xb.T @ diff) / n_samples + self.l2 * self.weights
            self.weights -= self.learning_rate * grad
        return self

    def predict_proba(self, X):
        Xb = np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float64)])
        logits = Xb @ self.weights
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def compute_metrics(y_true, y_pred):
    n_classes = len(LABEL_ORDER)
    conf = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf[t, p] += 1
    precision_scores = []
    recall_scores = []
    f1_scores = []
    for idx in range(n_classes):
        tp = conf[idx, idx]
        fp = conf[:, idx].sum() - tp
        fn = conf[idx, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
    accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precision_scores)),
        "macro_recall": float(np.mean(recall_scores)),
        "macro_f1": float(np.mean(f1_scores)),
        "confusion_matrix": conf.tolist(),
        "per_class_f1": {LABEL_ORDER[idx]: float(score) for idx, score in enumerate(f1_scores)},
    }


def draw_bar_chart(title, labels, values, output_path, color=(55, 114, 201)):
    width, height = 1100, 700
    margin_left, margin_bottom, margin_top, margin_right = 120, 180, 90, 60
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((margin_left, 25), title, fill="black")
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_value = max(values) if values else 1
    if max_value == 0:
        max_value = 1
    bar_width = max(20, int(plot_width / max(len(values), 1) * 0.65))
    spacing = max(10, int(plot_width / max(len(values), 1) * 0.35))
    base_y = margin_top + plot_height
    draw.line((margin_left, margin_top, margin_left, base_y), fill="black", width=2)
    draw.line((margin_left, base_y, width - margin_right, base_y), fill="black", width=2)
    for idx, (label, value) in enumerate(zip(labels, values)):
        x0 = margin_left + idx * (bar_width + spacing) + 10
        x1 = x0 + bar_width
        bar_height = int((value / max_value) * (plot_height - 20))
        y0 = base_y - bar_height
        draw.rectangle((x0, y0, x1, base_y), fill=color)
        draw.text((x0, y0 - 18), f"{value:.3f}" if isinstance(value, float) else str(value), fill="black")
        draw.text((x0, base_y + 10), label, fill="black")
    img.save(output_path)


def draw_confusion_matrix(title, matrix, labels, output_path):
    cell = 120
    margin = 160
    width = margin + cell * len(labels) + 80
    height = margin + cell * len(labels) + 100
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((margin, 25), title, fill="black")
    arr = np.asarray(matrix, dtype=np.float64)
    max_val = arr.max() if arr.size else 1.0
    if max_val == 0:
        max_val = 1.0
    for i, label in enumerate(labels):
        draw.text((margin + i * cell + 20, 90), label[:12], fill="black")
        draw.text((30, margin + i * cell + 40), label[:12], fill="black")
        for j in range(len(labels)):
            val = arr[i, j]
            intensity = int(255 - 180 * (val / max_val))
            color = (intensity, intensity, 255)
            x0 = margin + j * cell
            y0 = margin + i * cell
            x1 = x0 + cell - 4
            y1 = y0 + cell - 4
            draw.rectangle((x0, y0, x1, y1), fill=color, outline="black")
            draw.text((x0 + 45, y0 + 45), str(int(val)), fill="black")
    img.save(output_path)


def label_to_index(label: str) -> int:
    return LABEL_ORDER.index(label)


def split_rows(rows):
    result = {"train": [], "val": [], "test": []}
    for row in rows:
        result[row["split"]].append(row)
    return result


def latency_ms(bundle, model, texts):
    start = time.perf_counter()
    for text in texts:
        X = bundle.transform([text])
        _ = model.predict(X)
    elapsed = time.perf_counter() - start
    return (elapsed / max(len(texts), 1)) * 1000.0


def run_experiment(train_rows, val_rows, test_rows, config):
    train_texts = [row["caption"] for row in train_rows]
    val_texts = [row["caption"] for row in val_rows]
    test_texts = [row["caption"] for row in test_rows]
    y_train = np.asarray([label_to_index(row["label"]) for row in train_rows], dtype=np.int64)
    y_val = np.asarray([label_to_index(row["label"]) for row in val_rows], dtype=np.int64)
    y_test = np.asarray([label_to_index(row["label"]) for row in test_rows], dtype=np.int64)

    bundle = FeatureBundle(
        vectorizer=TfidfVectorizer(
            analyzer=config["analyzer"],
            ngram_range=config["ngram_range"],
            max_features=config["max_features"],
        ),
        use_extra_features=config["use_extra_features"],
    )
    X_train = bundle.fit_transform(train_texts)
    X_val = bundle.transform(val_texts)
    X_test = bundle.transform(test_texts)

    model = SoftmaxRegression(
        learning_rate=config["learning_rate"],
        epochs=config["epochs"],
        l2=config["l2"],
        class_weight=config["class_weight"],
    ).fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    val_metrics = compute_metrics(y_val, val_pred)
    test_metrics = compute_metrics(y_test, test_pred)
    avg_latency = latency_ms(bundle, model, test_texts or val_texts)

    errors = []
    probs = model.predict_proba(X_test)
    for row, true_idx, pred_idx, score_row in zip(test_rows, y_test, test_pred, probs):
        if true_idx == pred_idx:
            continue
        errors.append(
            {
                "id": row["id"],
                "caption": row["caption"],
                "true_label": LABEL_ORDER[true_idx],
                "predicted_label": LABEL_ORDER[pred_idx],
                "predicted_confidence": float(score_row[pred_idx]),
            }
        )

    return {
        "config": config,
        "bundle": bundle,
        "model": model,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "latency_ms": avg_latency,
        "test_errors": errors,
        "test_predictions": test_pred.tolist(),
    }


def save_json(path: Path, payload):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    ensure_dirs()
    rows = load_rows()
    split = split_rows(rows)
    train_rows = split["train"]
    val_rows = split["val"]
    test_rows = split["test"]

    label_counts = Counter(row["label"] for row in rows)
    split_counts = {name: len(items) for name, items in split.items()}
    draw_bar_chart(
        title="Caption class distribution for Lab 3 sample",
        labels=[label.replace("_", "\n") for label in LABEL_ORDER],
        values=[label_counts[label] for label in LABEL_ORDER],
        output_path=PLOTS_DIR / "lab3_class_distribution.png",
    )

    experiments = [
        {
            "name": "baseline_word_unigram",
            "analyzer": "word",
            "ngram_range": (1, 1),
            "max_features": 120,
            "use_extra_features": False,
            "class_weight": None,
            "learning_rate": 0.35,
            "epochs": 750,
            "l2": 1e-3,
        },
        {
            "name": "word_bigram_plus_features",
            "analyzer": "word",
            "ngram_range": (1, 2),
            "max_features": 180,
            "use_extra_features": True,
            "class_weight": None,
            "learning_rate": 0.30,
            "epochs": 850,
            "l2": 1e-3,
        },
        {
            "name": "char_ngram_balanced",
            "analyzer": "char",
            "ngram_range": (3, 5),
            "max_features": 220,
            "use_extra_features": True,
            "class_weight": "balanced",
            "learning_rate": 0.25,
            "epochs": 900,
            "l2": 5e-4,
        },
    ]

    log_path = EXPERIMENTS_DIR / "experiment_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    results = []
    for config in experiments:
        result = run_experiment(train_rows, val_rows, test_rows, config)
        results.append(result)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "name": config["name"],
                        "params": config,
                        "val_metrics": result["val_metrics"],
                        "test_metrics": result["test_metrics"],
                        "latency_ms": result["latency_ms"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    best = max(results, key=lambda item: item["val_metrics"]["macro_f1"])
    best_name = best["config"]["name"]

    draw_bar_chart(
        title="Validation macro F1 by experiment",
        labels=[res["config"]["name"].replace("_", "\n") for res in results],
        values=[res["val_metrics"]["macro_f1"] for res in results],
        output_path=PLOTS_DIR / "lab3_validation_macro_f1.png",
        color=(74, 144, 112),
    )
    draw_bar_chart(
        title="Test macro F1 by experiment",
        labels=[res["config"]["name"].replace("_", "\n") for res in results],
        values=[res["test_metrics"]["macro_f1"] for res in results],
        output_path=PLOTS_DIR / "lab3_test_macro_f1.png",
        color=(214, 110, 55),
    )
    draw_confusion_matrix(
        title=f"Confusion matrix for final model: {best_name}",
        matrix=best["test_metrics"]["confusion_matrix"],
        labels=LABEL_ORDER,
        output_path=PLOTS_DIR / "lab3_confusion_matrix.png",
    )

    error_breakdown = Counter(item["true_label"] for item in best["test_errors"])
    draw_bar_chart(
        title="Misclassified examples by true class",
        labels=[label.replace("_", "\n") for label in LABEL_ORDER],
        values=[error_breakdown.get(label, 0) for label in LABEL_ORDER],
        output_path=PLOTS_DIR / "lab3_error_breakdown.png",
        color=(154, 92, 180),
    )

    best_bundle = best["bundle"]
    best_model = best["model"]
    np.savez(
        MODELS_DIR / "final_caption_router_model.npz",
        weights=best_model.weights,
        idf=best_bundle.vectorizer.idf,
    )
    save_json(
        MODELS_DIR / "final_caption_router_model_meta.json",
        {
            "task": "caption_domain_routing",
            "label_order": LABEL_ORDER,
            "vectorizer": {
                "analyzer": best_bundle.vectorizer.analyzer,
                "ngram_range": best_bundle.vectorizer.ngram_range,
                "max_features": best_bundle.vectorizer.max_features,
                "vocabulary": list(best_bundle.vectorizer.vocab.keys()),
                "use_extra_features": best_bundle.use_extra_features,
            },
            "metrics": {
                "validation": best["val_metrics"],
                "test": best["test_metrics"],
                "latency_ms": best["latency_ms"],
            },
        },
    )
    save_json(EXPERIMENTS_DIR / "best_model_errors.json", best["test_errors"])
    save_json(
        ARTIFACTS_DIR / "lab3_summary.json",
        {
            "dataset_size": len(rows),
            "split_counts": split_counts,
            "label_counts": label_counts,
            "best_experiment": best_name,
            "best_validation_macro_f1": best["val_metrics"]["macro_f1"],
            "best_test_macro_f1": best["test_metrics"]["macro_f1"],
            "best_test_accuracy": best["test_metrics"]["accuracy"],
            "final_model_latency_ms": best["latency_ms"],
        },
    )

    print(json.dumps(
        {
            "dataset_size": len(rows),
            "split_counts": split_counts,
            "label_counts": label_counts,
            "best_experiment": best_name,
            "validation_macro_f1": round(best["val_metrics"]["macro_f1"], 4),
            "test_macro_f1": round(best["test_metrics"]["macro_f1"], 4),
            "test_accuracy": round(best["test_metrics"]["accuracy"], 4),
            "latency_ms": round(best["latency_ms"], 4),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
