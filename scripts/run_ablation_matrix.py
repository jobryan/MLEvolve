#!/usr/bin/env python3
"""Expand ablation variants, tasks, and seeds into stable run manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = ROOT / "configs/ablations/tasks/mle_bench_lite.json"
DEFAULT_VARIANTS = ROOT / "configs/ablations/variants.json"
DEFAULT_BUDGETS = ROOT / "configs/ablations/budgets.json"
DEFAULT_OUTPUT_ROOT = ROOT / ".context/ablation/run_manifests"
DEFAULT_SCREENING_MATRIX = ROOT / "configs/ablations/screening_matrix.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def stable_id(parts: dict[str, Any]) -> str:
    encoded = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def file_sha1(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slug(value: str) -> str:
    return value.replace("_", "-").replace("/", "-").replace(" ", "-").lower()


def resolve_repo_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def load_variant_config(variant: dict[str, Any]) -> dict[str, Any]:
    config_path = resolve_repo_path(variant.get("config_path"))
    if config_path is None or config_path.suffix != ".json" or not config_path.exists():
        return {}
    return load_json(config_path)


def select_tasks(tasks: list[dict[str, Any]], phase: str, task_filters: set[str]) -> list[dict[str, Any]]:
    selected = []
    for task in tasks:
        if task_filters and task["id"] not in task_filters:
            continue
        if phase == "smoke" and not task.get("smoke", False):
            continue
        selected.append(task)
    return selected


def select_variants(
    variants: list[dict[str, Any]],
    phase: str,
    system_filters: set[str],
    variant_filters: set[str],
    variant_specs: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    selected = []
    for variant in variants:
        identity = (variant["system"], variant["variant_id"])
        if variant_specs and identity not in variant_specs:
            continue
        if not variant_specs and system_filters and variant["system"] not in system_filters:
            continue
        if not variant_specs and variant_filters and variant["variant_id"] not in variant_filters:
            continue
        if phase not in variant.get("enabled_phases", []):
            continue
        selected.append(variant)
    return selected


def select_seeds(task_manifest: dict[str, Any], phase: str, seed_filters: set[int]) -> list[int]:
    if seed_filters:
        return sorted(seed_filters)
    seeds = task_manifest["defaults"]["phase_seed_policy"].get(phase)
    if not seeds:
        raise ValueError(f"No seed policy defined for phase {phase!r}")
    return list(seeds)


def build_manifest(
    task_manifest: dict[str, Any],
    variant_registry: dict[str, Any],
    budget_policy: dict[str, Any],
    task: dict[str, Any],
    variant: dict[str, Any],
    seed: int,
    phase: str,
    output_root: Path,
) -> dict[str, Any]:
    variant_config = load_variant_config(variant)
    config_path = resolve_repo_path(variant.get("config_path"))
    config_sha1 = file_sha1(config_path) if config_path is not None else None
    config_overrides = variant_config.get("overrides", {})
    identity = {
        "schema_version": "1.0",
        "task_manifest_version": task_manifest["manifest_version"],
        "variant_registry_version": variant_registry["variant_registry_version"],
        "variant_config_sha1": config_sha1,
        "benchmark_track": task_manifest["benchmark_track"],
        "system": variant["system"],
        "variant_id": variant["variant_id"],
        "task_id": task["id"],
        "phase": phase,
        "seed": seed,
    }
    suffix = stable_id(identity)
    run_id = f"{phase}-{slug(variant['system'])}-{slug(variant['variant_id'])}-{slug(task['id'])}-seed-{seed}-{suffix}"
    run_dir = output_root.parent / "runs" / run_id
    budget_profile = budget_policy["profiles"][phase]

    return {
        **identity,
        "run_id": run_id,
        "status": "planned",
        "task": {
            "id": task["id"],
            "name": task["name"],
            "domain": task["domain"],
            "difficulty": task["difficulty"],
            "resource_class": task["resource_class"],
            "metric_name": task["metric_name"],
            "metric_direction": task["metric_direction"],
            "expected_submission_path": task["expected_submission_path"],
            "prepare_command": task["prepare_command"],
            "grade_command": task["grade_command"],
            "grade_sample_command": task["grade_sample_command"],
        },
        "variant": {
            "summary": variant.get("summary"),
            "component_class": variant.get("component_class"),
            "config_path": variant.get("config_path"),
            "config_sha1": config_sha1,
            "research_question": variant_config.get("research_question"),
            "cli_flags": variant_config.get("cli_flags", []),
        },
        "config_overrides": config_overrides,
        "budget": {
            **budget_profile,
            "budget_policy_version": budget_policy["budget_policy_version"],
        },
        "resource_policy": {
            "resource_class": task["resource_class"],
            "matched_shared_benchmark": True,
        },
        "grader": {
            "name": task_manifest["defaults"]["grader_name"],
            "version": task_manifest["defaults"]["grader_version"],
            "command": task["grade_command"],
            "grade_sample_command": task["grade_sample_command"],
        },
        "artifacts": {
            "output_dir": str(run_dir.relative_to(ROOT)),
            "manifest_path": str((output_root / f"{run_id}.json").relative_to(ROOT)),
        },
    }


def iter_matrix(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    task_manifest = load_json(args.tasks)
    variant_registry = load_json(args.variants)
    budget_policy = load_json(args.budgets)

    tasks = select_tasks(task_manifest["tasks"], args.phase, set(args.task))
    variants = select_variants(
        variant_registry["variants"],
        args.phase,
        set(args.system),
        set(args.variant),
        set(getattr(args, "variant_specs", [])),
    )
    seeds = select_seeds(task_manifest, args.phase, set(args.seed))

    count = 0
    for variant in variants:
        for task in tasks:
            for seed in seeds:
                yield build_manifest(
                    task_manifest,
                    variant_registry,
                    budget_policy,
                    task,
                    variant,
                    seed,
                    args.phase,
                    args.output_root,
                )
                count += 1
                if args.max_runs is not None and count >= args.max_runs:
                    return


def write_manifest(manifest: dict[str, Any], force: bool) -> Path:
    path = ROOT / manifest["artifacts"]["manifest_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return path
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--variants", type=Path, default=DEFAULT_VARIANTS)
    parser.add_argument("--budgets", type=Path, default=DEFAULT_BUDGETS)
    parser.add_argument("--matrix", type=Path, help="Optional matrix JSON with task_ids, variant_ids, seeds, and phase.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--phase", choices=["smoke", "screening", "main", "confirmation"], default="smoke")
    parser.add_argument("--system", action="append", default=[], help="Filter by system; repeatable.")
    parser.add_argument("--variant", action="append", default=[], help="Filter by variant id; repeatable.")
    parser.add_argument("--task", action="append", default=[], help="Filter by task id; repeatable.")
    parser.add_argument("--seed", action="append", type=int, default=[], help="Filter by seed; repeatable.")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing manifests.")
    return parser.parse_args()


def apply_matrix_defaults(args: argparse.Namespace) -> None:
    args.variant_specs = []
    if args.matrix is None:
        return
    matrix = load_json(args.matrix)
    matrix_phase = matrix.get("phase")
    if matrix_phase and args.phase != matrix_phase:
        raise ValueError(f"matrix phase {matrix_phase!r} does not match --phase {args.phase!r}")
    if not args.task:
        args.task = list(matrix.get("task_ids", []))
    if not args.variant:
        args.variant = list(matrix.get("variant_ids", []))
    if not args.system and not args.variant and matrix.get("variant_specs"):
        args.variant_specs = [
            (str(spec["system"]), str(spec["variant_id"]))
            for spec in matrix["variant_specs"]
        ]
    if not args.seed:
        args.seed = [int(seed) for seed in matrix.get("seeds", [])]


def main() -> int:
    args = parse_args()
    apply_matrix_defaults(args)
    args.output_root = args.output_root.resolve()

    manifests = list(iter_matrix(args))
    if not manifests:
        print("No runs selected", file=sys.stderr)
        return 1

    for manifest in manifests:
        if args.dry_run:
            print(json.dumps(manifest, sort_keys=True))
        else:
            path = write_manifest(manifest, args.force)
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
