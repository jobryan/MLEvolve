#!/usr/bin/env python3
"""Validate AI Scientist v2 ablation variant config files and registry wiring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VARIANT_DIR = ROOT / "configs/ablations/ai_scientist_v2"
DEFAULT_REGISTRY = ROOT / "configs/ablations/variants.json"

REQUIRED_VARIANTS = {
    "stage_as_operator_default",
    "no_journal_memory",
    "semantic_scholar_off",
    "stage2_off",
    "draft_debug_improve_only",
    "creative_research_disabled",
    "ablation_disabled",
    "vlm_feedback_off",
    "writeup_review_off",
    "linear_stage",
    "greedy_selection",
    "debug_off_greedy",
    "diversity_ablated",
    "architecture_entropy",
    "novelty_filtering",
    "hardened_evaluator",
}

VALID_PHASES = {"smoke", "screening", "main", "confirmation", "native_finalist"}

SUPPORTED_OVERRIDE_TYPES: dict[str, type | tuple[type, ...]] = {
    "adapter.shared_benchmark": bool,
    "adapter.journal_memory": bool,
    "adapter.journal_memory_include_code": bool,
    "adapter.memory_backend": str,
    "adapter.semantic_scholar": bool,
    "adapter.vlm_feedback": bool,
    "adapter.writeup": bool,
    "adapter.review": bool,
    "adapter.stage2_enabled": bool,
    "adapter.stage3_enabled": bool,
    "adapter.stage4_enabled": bool,
    "adapter.selection_policy": str,
    "adapter.debug_enabled": bool,
    "adapter.diversity_mode": str,
    "adapter.novelty_filtering": bool,
    "adapter.evaluator_hardening": str,
    "adapter.audit_blocks_selection": bool,
    "agent.stages.stage2_max_iters": int,
    "agent.stages.stage3_max_iters": int,
    "agent.stages.stage4_max_iters": int,
    "agent.search.debug_prob": (int, float),
    "agent.search.max_debug_depth": int,
}

ENUMS: dict[str, set[Any]] = {
    "adapter.selection_policy": {
        "default_bfts",
        "linear_stage",
        "greedy",
        "debug_off_greedy",
        "uct_mcts",
        "mcgs_lite",
    },
    "adapter.memory_backend": {"llm_summary", "retrieval"},
    "adapter.diversity_mode": {"default", "none", "architecture_entropy", "novelty"},
    "adapter.evaluator_hardening": {"standard", "strict"},
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def matches_type(value: Any, expected: type | tuple[type, ...]) -> bool:
    if expected is bool:
        return isinstance(value, bool)
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
        "cli_flags",
        "overrides",
    ]:
        if field not in data:
            errors.append(f"{path}: missing required field {field!r}")

    if data.get("system") != "ai_scientist_v2":
        errors.append(f"{path}: system must be 'ai_scientist_v2'")

    if data.get("variant_id") != path.stem:
        errors.append(f"{path}: variant_id must match filename stem {path.stem!r}")

    phases = data.get("enabled_phases", [])
    if not isinstance(phases, list) or not phases:
        errors.append(f"{path}: enabled_phases must be a non-empty list")
    else:
        unknown_phases = sorted(set(phases) - VALID_PHASES)
        if unknown_phases:
            errors.append(f"{path}: unsupported phases {unknown_phases}")

    cli_flags = data.get("cli_flags", [])
    if not isinstance(cli_flags, list) or any(not isinstance(flag, str) for flag in cli_flags):
        errors.append(f"{path}: cli_flags must be a string list")

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

    if overrides.get("adapter.writeup") is False and "--skip_writeup" not in cli_flags:
        errors.append(f"{path}: adapter.writeup=false requires --skip_writeup")
    if overrides.get("adapter.review") is False and "--skip_review" not in cli_flags:
        errors.append(f"{path}: adapter.review=false requires --skip_review")

    return variant_id, errors


def validate_registry(registry_path: Path, variant_files: dict[str, Path]) -> list[str]:
    errors: list[str] = []
    try:
        registry = load_json(registry_path)
    except Exception as exc:
        return [f"{registry_path}: cannot parse JSON: {exc}"]

    registry_entries = registry.get("variants", [])
    ais_entries = {
        entry.get("variant_id"): entry
        for entry in registry_entries
        if entry.get("system") == "ai_scientist_v2"
    }

    missing_from_registry = sorted(set(variant_files) - set(ais_entries))
    if missing_from_registry:
        errors.append(f"{registry_path}: missing AI Scientist v2 entries {missing_from_registry}")

    missing_files = sorted(set(ais_entries) - set(variant_files))
    if missing_files:
        errors.append(f"{registry_path}: AI Scientist v2 entries without config files {missing_files}")

    for variant_id, path in variant_files.items():
        entry = ais_entries.get(variant_id)
        if not entry:
            continue
        expected = path.relative_to(ROOT).as_posix()
        if entry.get("config_path") != expected:
            errors.append(
                f"{registry_path}: {variant_id} config_path must be {expected!r}, got {entry.get('config_path')!r}"
            )

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant-dir", type=Path, default=DEFAULT_VARIANT_DIR)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--skip-registry", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    variant_paths = sorted(path for path in args.variant_dir.glob("*.json") if path.is_file())
    variant_files = {path.stem: path for path in variant_paths}
    errors: list[str] = []

    missing_required = sorted(REQUIRED_VARIANTS - set(variant_files))
    if missing_required:
        errors.append(f"{args.variant_dir}: missing required variants {missing_required}")

    for path in variant_paths:
        _, variant_errors = validate_variant(path)
        errors.extend(variant_errors)

    if not args.skip_registry:
        errors.extend(validate_registry(args.registry, variant_files))

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(f"validated {len(variant_paths)} AI Scientist v2 ablation configs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
