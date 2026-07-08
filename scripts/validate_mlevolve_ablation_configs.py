#!/usr/bin/env python3
"""Validate MLEvolve ablation variant override files and registry wiring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VARIANT_DIR = ROOT / "configs/ablations/mlevolve"
DEFAULT_REGISTRY = ROOT / "configs/ablations/variants.json"

RECOMMENDED_VARIANTS = {
    "default_mcgs",
    "no_memory",
    "child_history_only",
    "global_retrieval",
    "linear_chain",
    "greedy_tree",
    "vanilla_mcts",
    "no_fusion_evolution",
    "single_shot",
    "stepwise_diff",
    "no_cold_start",
    "diversity_ablated",
    "diversity_high",
    "novelty_lambda_005",
    "novelty_lambda_010",
    "novelty_lambda_020",
    "novelty_lambda_040",
    "all_strong",
    "strong_code_cheap_feedback",
}

VALID_PHASES = {"smoke", "screening", "main", "confirmation", "native_finalist"}

SUPPORTED_OVERRIDE_TYPES: dict[str, type | tuple[type, ...]] = {
    "ablation.enabled": bool,
    "ablation.variant_id": str,
    "ablation.search_policy": str,
    "ablation.child_memory": bool,
    "ablation.global_memory_filter": str,
    "ablation.dissimilar_guidance": bool,
    "ablation.diversity_mode": str,
    "ablation.novelty_lambda": (int, float),
    "ablation.novelty_in_reward": bool,
    "ablation.operator_set": str,
    "ablation.model_profile": str,
    "agent.steps": int,
    "agent.initial_drafts": int,
    "agent.code.model": str,
    "agent.feedback.model": str,
    "agent.check_data_leakage": bool,
    "agent.use_code_review": bool,
    "agent.use_global_memory": bool,
    "agent.use_diff_mode": bool,
    "agent.use_stepwise_generation": bool,
    "agent.use_evolution": bool,
    "agent.use_fusion": bool,
    "agent.use_aggregation": bool,
    "agent.search.num_drafts": int,
    "agent.search.parallel_search_num": int,
    "agent.search.use_stagnation_detection": bool,
    "agent.search.explore_switch_start": (int, float),
    "agent.search.explore_switch_end": (int, float),
    "agent.search.min_exploration_weight": (int, float),
    "coldstart.use_coldstart": bool,
}

ENUMS: dict[str, set[Any]] = {
    "ablation.search_policy": {"mcgs", "linear_chain", "greedy_tree", "vanilla_mcts", "progressive_mcts", "single_shot"},
    "ablation.global_memory_filter": {"all", "none", "success_only", "failure_only"},
    "ablation.diversity_mode": {"default", "none", "high", "novelty"},
    "ablation.operator_set": {"full", "draft_debug_improve", "no_fusion_evolution", "draft_only"},
    "ablation.model_profile": {"default", "all_strong", "strong_code_cheap_feedback"},
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def matches_type(value: Any, expected: type | tuple[type, ...]) -> bool:
    if expected is bool:
        return is_bool(value)
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == (int, float):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, expected)


def validate_variant(path: Path) -> tuple[str, list[str]]:
    errors: list[str] = []
    try:
        data = load_json(path)
    except Exception as exc:
        return path.stem, [f"{path}: cannot parse JSON: {exc}"]

    variant_id = str(data.get("variant_id", path.stem))
    for field in [
        "variant_id",
        "system",
        "component_class",
        "summary",
        "research_question",
        "inherits",
        "enabled_phases",
        "overrides",
    ]:
        if field not in data:
            errors.append(f"{path}: missing required field {field!r}")

    if data.get("system") != "mlevolve":
        errors.append(f"{path}: system must be 'mlevolve'")

    if data.get("variant_id") != path.stem:
        errors.append(f"{path}: variant_id must match filename stem {path.stem!r}")

    phases = data.get("enabled_phases", [])
    if not isinstance(phases, list) or not phases:
        errors.append(f"{path}: enabled_phases must be a non-empty list")
    else:
        unknown_phases = sorted(set(phases) - VALID_PHASES)
        if unknown_phases:
            errors.append(f"{path}: unsupported phases {unknown_phases}")

    overrides = data.get("overrides", {})
    if not isinstance(overrides, dict) or not overrides:
        errors.append(f"{path}: overrides must be a non-empty object")
        return variant_id, errors

    for key, value in overrides.items():
        expected = SUPPORTED_OVERRIDE_TYPES.get(key)
        if expected is None:
            errors.append(f"{path}: unsupported override key {key!r}")
            continue
        if not matches_type(value, expected):
            errors.append(f"{path}: override {key!r} has invalid type {type(value).__name__}")
        accepted = ENUMS.get(key)
        if accepted is not None and value not in accepted:
            errors.append(f"{path}: override {key!r} has unsupported value {value!r}")

    static_variant = overrides.get("ablation.variant_id")
    if static_variant is not None and static_variant != data.get("variant_id"):
        errors.append(f"{path}: ablation.variant_id override must match variant_id")

    novelty = overrides.get("ablation.novelty_lambda")
    if novelty is not None and float(novelty) < 0:
        errors.append(f"{path}: ablation.novelty_lambda must be non-negative")

    return variant_id, errors


def validate_registry(registry_path: Path, variant_files: dict[str, Path]) -> list[str]:
    errors: list[str] = []
    try:
        registry = load_json(registry_path)
    except Exception as exc:
        return [f"{registry_path}: cannot parse JSON: {exc}"]

    registry_entries = registry.get("variants", [])
    if not isinstance(registry_entries, list):
        return [f"{registry_path}: variants must be a list"]

    mlevolve_entries = {
        entry.get("variant_id"): entry
        for entry in registry_entries
        if entry.get("system") == "mlevolve"
    }

    missing_from_registry = sorted(set(variant_files) - set(mlevolve_entries))
    if missing_from_registry:
        errors.append(f"{registry_path}: missing MLEvolve registry entries {missing_from_registry}")

    missing_files = sorted(set(mlevolve_entries) - set(variant_files))
    if missing_files:
        errors.append(f"{registry_path}: MLEvolve registry entries without config files {missing_files}")

    for variant_id, path in variant_files.items():
        entry = mlevolve_entries.get(variant_id)
        if not entry:
            continue
        config_path = entry.get("config_path")
        expected = path.relative_to(ROOT).as_posix()
        if config_path != expected:
            errors.append(
                f"{registry_path}: {variant_id} config_path must be {expected!r}, got {config_path!r}"
            )

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant-dir", type=Path, default=DEFAULT_VARIANT_DIR)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--skip-registry",
        action="store_true",
        help="Validate variant files only; useful for isolated negative tests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    variant_paths = sorted(path for path in args.variant_dir.glob("*.json") if path.is_file())
    variant_files = {path.stem: path for path in variant_paths}
    errors: list[str] = []

    missing_recommended = sorted(RECOMMENDED_VARIANTS - set(variant_files))
    if missing_recommended:
        errors.append(f"{args.variant_dir}: missing recommended variants {missing_recommended}")

    for path in variant_paths:
        _, variant_errors = validate_variant(path)
        errors.extend(variant_errors)

    if not args.skip_registry:
        errors.extend(validate_registry(args.registry, variant_files))

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(f"validated {len(variant_paths)} MLEvolve ablation configs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
