"""Deterministic diversity labels and novelty scoring for ablation analysis."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any


KEYWORD_LABELS: dict[str, dict[str, list[str]]] = {
    "model_family": {
        "tree_boosting": ["xgboost", "lightgbm", "catboost", "random forest", "gbdt"],
        "linear": ["linear regression", "logistic regression", "ridge", "lasso", "svm"],
        "neural_network": ["cnn", "rnn", "lstm", "gru", "transformer", "neural", "resnet", "efficientnet"],
        "nearest_neighbor": ["knn", "nearest neighbor"],
    },
    "architecture": {
        "ensemble": ["ensemble", "blend", "stacking", "bagging", "voting"],
        "pretrained": ["pretrained", "fine-tune", "finetune", "foundation model", "backbone"],
        "feature_extractor": ["feature extractor", "frozen", "embeddings"],
        "baseline": ["baseline", "simple"],
    },
    "data_strategy": {
        "augmentation": ["augmentation", "augment", "flip", "crop", "mixup", "cutmix"],
        "pseudo_labeling": ["pseudo-label", "pseudo label", "self-training"],
        "cross_validation": ["cross validation", "kfold", "stratified"],
        "resampling": ["oversample", "undersample", "class weight", "imbalance"],
    },
    "feature_strategy": {
        "text_features": ["tf-idf", "ngram", "token", "embedding", "bert"],
        "image_features": ["image", "pixel", "cnn", "augmentation"],
        "tabular_features": ["feature engineering", "categorical", "target encoding", "scaling"],
        "time_features": ["time", "date", "seasonality", "lag"],
    },
    "training_strategy": {
        "hyperparameter_tuning": ["hyperparameter", "learning rate", "scheduler", "grid search", "bayesian"],
        "regularization": ["dropout", "weight decay", "early stopping", "regularization"],
        "loss_change": ["loss", "focal", "label smoothing", "objective"],
        "calibration": ["calibration", "threshold", "postprocess", "post-process"],
    },
    "ensembling": {
        "none": ["single model"],
        "averaging": ["average", "mean ensemble"],
        "stacking": ["stacking", "meta model"],
        "voting": ["vote", "voting"],
    },
}


TOKEN_RE = re.compile(r"[a-zA-Z0-9_+-]+")


def normalize_text(text: str | None) -> str:
    return (text or "").lower()


def extract_strategy_labels(plan_text: str | None, code_text: str | None = None) -> dict[str, str]:
    text = normalize_text(f"{plan_text or ''}\n{code_text or ''}")
    labels: dict[str, str] = {}
    for category, choices in KEYWORD_LABELS.items():
        labels[category] = "unspecified"
        for label, keywords in choices.items():
            if any(keyword in text for keyword in keywords):
                labels[category] = label
                break
    return labels


def tokenize(text: str | None) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))


def hashed_embedding(text: str | None, dims: int = 64) -> list[float]:
    vector = [0.0] * dims
    for token in tokenize(text):
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dims
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine_distance(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 1.0
    dot = sum(a * b for a, b in zip(left, right))
    return max(0.0, min(2.0, 1.0 - dot))


def nearest_neighbor_distance(text: str | None, previous_texts: list[str], dims: int = 64) -> float | None:
    if not previous_texts:
        return None
    current = hashed_embedding(text, dims=dims)
    distances = [cosine_distance(current, hashed_embedding(previous, dims=dims)) for previous in previous_texts]
    return min(distances) if distances else None


def node_summary_text(node: Any) -> str:
    parts = [
        getattr(node, "plan", None),
        getattr(node, "code_summary", None),
    ]
    if not any(parts):
        parts.append(getattr(node, "code", None))
    return "\n".join(str(part) for part in parts if part)


def previous_node_texts(nodes: list[Any], current_node: Any | None = None) -> list[str]:
    texts = []
    current_id = getattr(current_node, "id", None)
    for node in nodes:
        if current_id is not None and getattr(node, "id", None) == current_id:
            continue
        text = node_summary_text(node)
        if text.strip():
            texts.append(text)
    return texts


def summarize_node_diversity(node: Any, previous_nodes: list[Any] | None = None) -> dict[str, Any]:
    text = node_summary_text(node)
    previous_text = previous_node_texts(previous_nodes or [], current_node=node)
    return {
        "strategy_labels": extract_strategy_labels(text, getattr(node, "code", None)),
        "embedding": hashed_embedding(text),
        "nearest_neighbor_distance": nearest_neighbor_distance(text, previous_text),
    }
