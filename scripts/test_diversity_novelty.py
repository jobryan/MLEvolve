#!/usr/bin/env python3
"""Checks for deterministic diversity labels and novelty scoring."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import contextlib
import os
from types import SimpleNamespace

from agents.workflow_controls import diversity_prompts_enabled
import utils.diversity_novelty as dn
from utils.diversity_novelty import (
    HASH_BACKEND_LABEL,
    extract_strategy_labels,
    get_embedding,
    hashed_embedding,
    nearest_neighbor_distance,
    resolve_embedding_backend,
    summarize_node_diversity,
)


def test_strategy_labels() -> None:
    labels = extract_strategy_labels(
        "Fine-tune a pretrained EfficientNet CNN with image augmentation and an ensemble.",
        "use dropout and early stopping",
    )
    assert labels["model_family"] == "neural_network"
    assert labels["architecture"] == "ensemble"
    assert labels["data_strategy"] == "augmentation"
    assert labels["training_strategy"] == "regularization"


def test_repeated_plan_less_novel_than_distinct_plan() -> None:
    previous = ["Use xgboost with tabular feature engineering and cross validation"]
    repeated = nearest_neighbor_distance(
        "Use xgboost with tabular feature engineering and cross validation",
        previous,
    )
    distinct = nearest_neighbor_distance(
        "Fine-tune a pretrained transformer with image augmentation and an ensemble",
        previous,
    )
    assert repeated is not None
    assert distinct is not None
    assert repeated < distinct


def _fake_agent(diversity_mode: str | None) -> SimpleNamespace:
    return SimpleNamespace(cfg=SimpleNamespace(ablation=SimpleNamespace(diversity_mode=diversity_mode)))


def test_diversity_prompts_gate() -> None:
    assert diversity_prompts_enabled(_fake_agent("default")) is True
    assert diversity_prompts_enabled(_fake_agent(None)) is True
    assert diversity_prompts_enabled(_fake_agent("none")) is False
    # Agents without an ablation config keep legacy behavior.
    assert diversity_prompts_enabled(SimpleNamespace(cfg=SimpleNamespace())) is True


def test_diversity_prompt_blocks_gated_in_agent_sources() -> None:
    # Source-level check (agents import torch/omegaconf, unavailable in the
    # local harness environment): the diversity block must sit behind the gate.
    checks = {
        "agents/draft_agent.py": "NOVELTY & DIVERSITY REQUIREMENT",
        "agents/improve_agent.py": "distinctly different from existing attempts",
        "agents/evolution_agent.py": "distinctly different from existing attempts",
    }
    for rel_path, marker in checks.items():
        source = (ROOT / rel_path).read_text(encoding="utf-8")
        assert marker in source, f"{rel_path}: diversity marker missing"
        gate_pos = source.find("if diversity_prompts_enabled(agent):")
        assert gate_pos != -1, f"{rel_path}: diversity block must be gated"
        assert gate_pos < source.find(marker), f"{rel_path}: gate must precede the block"


@contextlib.contextmanager
def _env_backend(value: str | None):
    previous = os.environ.get("MLEVOLVE_EMBEDDING_BACKEND")
    if value is None:
        os.environ.pop("MLEVOLVE_EMBEDDING_BACKEND", None)
    else:
        os.environ["MLEVOLVE_EMBEDDING_BACKEND"] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("MLEVOLVE_EMBEDDING_BACKEND", None)
        else:
            os.environ["MLEVOLVE_EMBEDDING_BACKEND"] = previous


@contextlib.contextmanager
def _patched_loader(loader):
    original = dn._load_semantic_model
    dn._load_semantic_model = loader
    dn._reset_semantic_state()
    try:
        yield
    finally:
        dn._load_semantic_model = original
        dn._reset_semantic_state()


class _StubEmbeddingModel:
    """Fake memory-stack model; never touches sentence-transformers/torch."""

    model_name = "stub-org/stub-model"

    def encode(self, texts):
        return [[float(len(text)), 1.0, 0.0] for text in texts]


def test_backend_selection_env_driven() -> None:
    with _env_backend(None):
        assert resolve_embedding_backend() == "hash"
        assert resolve_embedding_backend("semantic") == "semantic"
    with _env_backend("semantic"):
        assert resolve_embedding_backend() == "semantic"
        # Explicit argument overrides the env.
        assert resolve_embedding_backend("hash") == "hash"
    with _env_backend("bogus-backend"):
        assert resolve_embedding_backend() == "hash"
    # Default behavior (env unset) must remain byte-identical to the old hash path.
    with _env_backend(None):
        vector, label = get_embedding("use xgboost with cross validation")
        assert label == HASH_BACKEND_LABEL
        assert vector == hashed_embedding("use xgboost with cross validation")


def test_semantic_backend_uses_memory_stack_model_and_label() -> None:
    with _patched_loader(lambda: _StubEmbeddingModel()):
        vector, label = get_embedding("fine-tune a pretrained transformer", backend="semantic")
        assert label == "semantic-stub-model-v1"
        assert len(vector) == 3
        norm = sum(value * value for value in vector) ** 0.5
        assert abs(norm - 1.0) < 1e-9  # L2-normalized
        # nearest_neighbor_distance honors the semantic backend too.
        distance = nearest_neighbor_distance("same text", ["same text"], backend="semantic")
        assert distance is not None and distance < 1e-9


def test_semantic_loader_failure_falls_back_to_hash() -> None:
    def _boom():
        raise ImportError("torch not installed")

    with _patched_loader(_boom):
        vector, label = get_embedding("use xgboost", backend="semantic")
        assert label == HASH_BACKEND_LABEL
        assert vector == hashed_embedding("use xgboost")
        # Subsequent calls keep working without retry crashes.
        distance = nearest_neighbor_distance("a plan", ["another plan"], backend="semantic")
        assert distance is not None
        assert dn._SEMANTIC["failed"] is True


def test_summarize_node_diversity_label_propagation() -> None:
    node = SimpleNamespace(id="n1", plan="Use xgboost with cross validation", code=None, code_summary=None)
    other = SimpleNamespace(id="n0", plan="Fine-tune a pretrained transformer", code=None, code_summary=None)

    with _env_backend(None):
        summary = summarize_node_diversity(node, [other])
        assert summary["embedding_backend"] == HASH_BACKEND_LABEL
        assert summary["nearest_neighbor_distance"] is not None
        assert "strategy_labels" in summary

    with _patched_loader(lambda: _StubEmbeddingModel()):
        with _env_backend("semantic"):
            summary = summarize_node_diversity(node, [other])
            assert summary["embedding_backend"] == "semantic-stub-model-v1"
            assert len(summary["embedding"]) == 3

    # Fallback keeps the label honest: hash label when semantic is unavailable.
    def _boom():
        raise RuntimeError("no model weights")

    with _patched_loader(_boom):
        with _env_backend("semantic"):
            summary = summarize_node_diversity(node, [other])
            assert summary["embedding_backend"] == HASH_BACKEND_LABEL


def test_no_gpu_env_forces_cpu_device() -> None:
    previous = os.environ.get("MLEVOLVE_NO_GPU")
    os.environ["MLEVOLVE_NO_GPU"] = "1"
    try:
        model_path, device = dn._read_memory_embedding_config()
        assert device == "cpu"
        assert model_path  # model path comes from config.yaml (memory stack)
    finally:
        if previous is None:
            os.environ.pop("MLEVOLVE_NO_GPU", None)
        else:
            os.environ["MLEVOLVE_NO_GPU"] = previous


def main() -> int:
    test_strategy_labels()
    test_repeated_plan_less_novel_than_distinct_plan()
    test_diversity_prompts_gate()
    test_diversity_prompt_blocks_gated_in_agent_sources()
    test_backend_selection_env_driven()
    test_semantic_backend_uses_memory_stack_model_and_label()
    test_semantic_loader_failure_falls_back_to_hash()
    test_summarize_node_diversity_label_propagation()
    test_no_gpu_env_forces_cpu_device()
    print("diversity novelty tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
