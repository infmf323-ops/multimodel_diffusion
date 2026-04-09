import hashlib
import csv
import json
import math
import re
from pathlib import Path

import numpy as np

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
    "person_token",
    "celebrities",
    "celebrity",
    "bride",
    "groom",
    "premiere",
    "stage",
    "singer",
    "star",
}
NATURE_KEYWORDS = {
    "animal_token",
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


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("sports team", "sports_team")
    text = text.replace("'", "")
    text = re.sub(r"[^a-z0-9_!?.,\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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


class CaptionRouterModel:
    def __init__(self, model_npz_path: Path, meta_path: Path, sample_data_path: Path | None = None):
        self.model_npz_path = model_npz_path
        self.meta_path = meta_path
        self.sample_data_path = sample_data_path

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        weights = np.load(model_npz_path)

        self.weights = weights["weights"]
        self.idf = weights["idf"]
        self.label_order = meta["label_order"]
        self.vectorizer = meta["vectorizer"]
        self.vocab = {token: idx for idx, token in enumerate(self.vectorizer["vocabulary"])}
        self.model_sha = hashlib.sha256(model_npz_path.read_bytes()).hexdigest()[:12]
        self.meta = meta
        self.extra_mean, self.extra_std = self._load_extra_feature_stats()

    def _char_ngrams(self, text: str):
        analyzer_text = f" {model_text(text)} "
        ngram_min, ngram_max = self.vectorizer["ngram_range"]
        ngrams = []
        for n in range(ngram_min, ngram_max + 1):
            for idx in range(len(analyzer_text) - n + 1):
                ngrams.append(analyzer_text[idx : idx + n])
        return ngrams

    def _vectorize(self, texts):
        matrix = np.zeros((len(texts), len(self.vocab)), dtype=np.float64)
        for row_idx, text in enumerate(texts):
            counts = {}
            for token in self._char_ngrams(text):
                counts[token] = counts.get(token, 0) + 1
            total = sum(counts.values()) or 1
            for token, count in counts.items():
                col_idx = self.vocab.get(token)
                if col_idx is None:
                    continue
                tf = count / total
                matrix[row_idx, col_idx] = tf * self.idf[col_idx]
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms

        if self.vectorizer.get("use_extra_features"):
            extra = self._extra_features(texts)
            extra = (extra - self.extra_mean) / self.extra_std
            matrix = np.hstack([matrix, extra])

        return matrix

    def _raw_extra_features(self, text: str):
        normalized = model_text(text)
        tokens = re.findall(r"[a-z_]+", normalized)
        return [
            len(tokens),
            len(normalized),
            normalized.count("!"),
            int("sports_team" in normalized),
            int("person" in tokens),
            int(any(token in SPORTS_KEYWORDS for token in tokens)),
            int(any(token in PEOPLE_KEYWORDS for token in tokens)),
            int(any(token in NATURE_KEYWORDS for token in tokens)),
        ]

    def _extra_features(self, texts):
        return np.asarray([self._raw_extra_features(text) for text in texts], dtype=np.float64)

    def _load_extra_feature_stats(self):
        if not self.vectorizer.get("use_extra_features"):
            return np.zeros((1, 0), dtype=np.float64), np.ones((1, 0), dtype=np.float64)
        if not self.sample_data_path or not self.sample_data_path.exists():
            return np.zeros((1, 8), dtype=np.float64), np.ones((1, 8), dtype=np.float64)

        captions = []
        with self.sample_data_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                captions.append(row["caption"])

        features = self._extra_features(captions)
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        return mean, std

    def predict(self, caption: str):
        features = self._vectorize([caption])
        bias = np.ones((features.shape[0], 1), dtype=np.float64)
        logits = np.hstack([features, bias]) @ self.weights
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        prob_row = probs[0]
        top_idx = int(prob_row.argmax())

        return {
            "predicted_label": self.label_order[top_idx],
            "confidence": float(prob_row[top_idx]),
            "probabilities": {
                label: float(prob_row[idx]) for idx, label in enumerate(self.label_order)
            },
            "model_sha": self.model_sha,
            "vectorizer_type": self.vectorizer["analyzer"],
        }

    def info(self):
        return {
            "task": self.meta["task"],
            "labels": self.label_order,
            "vectorizer": self.vectorizer,
            "model_sha": self.model_sha,
            "metrics": self.meta["metrics"],
        }
