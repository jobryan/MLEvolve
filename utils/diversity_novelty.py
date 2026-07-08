"""Deterministic diversity labels and novelty scoring for ablation analysis."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger("MLEvolve")

HASH_BACKEND_LABEL = "hashing-token-v1"
EMBEDDING_BACKENDS = ("hash", "semantic")
DEFAULT_SEMANTIC_MODEL_PATH = "BAAI/bge-base-en-v1.5"

# Lazy singleton state for the semantic (memory-stack) embedding model.
_SEMANTIC = {"model": None, "label": None, "failed": False, "warned": False}


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


def _read_memory_embedding_config() -> tuple[str, str]:
    """Read the memory-stack embedding model path/device from config/config.yaml.

    Uses a dependency-light line scan so this module stays stdlib-only; falls
    back to the shipped defaults when the config file is unreadable.
    """
    model_path = DEFAULT_SEMANTIC_MODEL_PATH
    device = "cpu"
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    try:
        text = config_path.read_text(encoding="utf-8")
        model_match = re.search(
            r"^\s*memory_embedding_model_path:\s*[\"']?([^\"'#\n]+)", text, re.MULTILINE
        )
        if model_match:
            model_path = model_match.group(1).strip()
        device_match = re.search(
            r"^\s*memory_embedding_device:\s*[\"']?([^\"'#\n]+)", text, re.MULTILINE
        )
        if device_match:
            device = device_match.group(1).strip()
    except OSError:
        pass
    if os.environ.get("MLEVOLVE_NO_GPU") == "1":
        device = "cpu"
    return model_path, device


def semantic_model_label(model_path: str) -> str:
    name = (model_path or "unknown").rstrip("/").split("/")[-1] or "unknown"
    return f"semantic-{name}-v1"


def _load_semantic_model() -> Any:
    """Lazily load the SAME embedding model the global-memory stack uses.

    Reuses agents.memory.embedding_models.EmbeddingModel (sentence-transformers)
    so weights/config come from one place. Import is deferred so environments
    without torch never pay for it unless the semantic backend is requested.
    """
    from agents.memory.embedding_models import EmbeddingModel  # heavy import, keep lazy

    model_path, device = _read_memory_embedding_config()
    return EmbeddingModel(model_type="local", model_name=model_path, device=device)


def _reset_semantic_state() -> None:
    """Test hook: clear the cached semantic model / failure flag."""
    _SEMANTIC.update({"model": None, "label": None, "failed": False, "warned": False})


def _semantic_model() -> tuple[Any, str]:
    if _SEMANTIC["failed"]:
        raise RuntimeError("semantic embedding backend previously failed to load")
    if _SEMANTIC["model"] is None:
        model = _load_semantic_model()
        model_name = getattr(model, "model_name", None) or _read_memory_embedding_config()[0]
        _SEMANTIC["model"] = model
        _SEMANTIC["label"] = semantic_model_label(str(model_name))
    return _SEMANTIC["model"], _SEMANTIC["label"]


def _mark_semantic_failed(exc: Exception) -> None:
    _SEMANTIC["failed"] = True
    if not _SEMANTIC["warned"]:
        _SEMANTIC["warned"] = True
        logger.warning(
            f"[diversity] semantic embedding backend unavailable ({exc!r}); "
            f"falling back to {HASH_BACKEND_LABEL}"
        )


def semantic_embedding(text: str | None) -> list[float]:
    """Embed text with the memory-stack sentence-embedding model (L2-normalized).

    Raises if the model cannot be loaded; use get_embedding for the safe,
    hash-fallback entry point.
    """
    model, _ = _semantic_model()
    raw = model.encode([normalize_text(text) or ""])[0]
    vector = [float(value) for value in raw]
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def resolve_embedding_backend(backend: str | None = None) -> str:
    """Resolve the requested backend; env MLEVOLVE_EMBEDDING_BACKEND, default hash."""
    requested = (backend or os.environ.get("MLEVOLVE_EMBEDDING_BACKEND") or "hash").strip().lower()
    if requested not in EMBEDDING_BACKENDS:
        logger.warning(
            f"[diversity] unknown embedding backend {requested!r}; falling back to 'hash'"
        )
        return "hash"
    return requested


def _effective_backend(backend: str | None = None) -> tuple[str, str]:
    """Return (backend, label) actually usable right now, degrading to hash."""
    requested = resolve_embedding_backend(backend)
    if requested == "semantic":
        try:
            _, label = _semantic_model()
            return "semantic", label
        except Exception as exc:  # model unavailable / import error: never crash a run
            _mark_semantic_failed(exc)
    return "hash", HASH_BACKEND_LABEL


def get_embedding(text: str | None, backend: str | None = None) -> tuple[list[float], str]:
    """Embed text with the configured backend. Returns (vector, backend_label).

    backend in {"hash", "semantic"}; default comes from MLEVOLVE_EMBEDDING_BACKEND
    (default "hash"). Any semantic failure falls back to the hash backend with a
    logged warning.
    """
    effective, label = _effective_backend(backend)
    if effective == "semantic":
        try:
            return semantic_embedding(text), label
        except Exception as exc:
            _mark_semantic_failed(exc)
    return hashed_embedding(text), HASH_BACKEND_LABEL


def cosine_distance(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 1.0
    dot = sum(a * b for a, b in zip(left, right))
    return max(0.0, min(2.0, 1.0 - dot))


def nearest_neighbor_distance(
    text: str | None,
    previous_texts: list[str],
    dims: int = 64,
    backend: str | None = None,
) -> float | None:
    if not previous_texts:
        return None
    effective, _ = _effective_backend(backend)
    if effective == "semantic":
        try:
            current = semantic_embedding(text)
            distances = [
                cosine_distance(current, semantic_embedding(previous))
                for previous in previous_texts
            ]
            return min(distances) if distances else None
        except Exception as exc:
            _mark_semantic_failed(exc)
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


def summarize_node_diversity(
    node: Any,
    previous_nodes: list[Any] | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    text = node_summary_text(node)
    previous_text = previous_node_texts(previous_nodes or [], current_node=node)
    embedding, backend_label = get_embedding(text, backend=backend)
    # Pin the distance computation to the backend that actually produced the
    # embedding so the summary is internally consistent even after a fallback.
    distance_backend = "semantic" if backend_label != HASH_BACKEND_LABEL else "hash"
    return {
        "strategy_labels": extract_strategy_labels(text, getattr(node, "code", None)),
        "embedding": embedding,
        "embedding_backend": backend_label,
        "nearest_neighbor_distance": nearest_neighbor_distance(
            text, previous_text, backend=distance_backend
        ),
    }
