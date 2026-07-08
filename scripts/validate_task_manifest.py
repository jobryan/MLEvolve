#!/usr/bin/env python3
"""Validate ablation task manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs/ablations/tasks/mle_bench_lite.json"
REQUIRED_TOP_LEVEL = {
    "manifest_version",
    "benchmark_track",
    "source",
    "defaults",
    "required_task_fields",
    "tasks",
}
ALLOWED_DIRECTIONS = {"minimize", "maximize"}


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("manifest must be a JSON object")
    return data


def is_unresolved(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        stripped = value.strip().lower()
        return stripped == "" or stripped in {"todo", "tbd", "unknown_required"}
    return False


def validate_manifest(path: Path) -> list[str]:
    data = load_manifest(path)
    errors: list[str] = []

    missing_top = sorted(REQUIRED_TOP_LEVEL - data.keys())
    for field in missing_top:
        errors.append(f"missing top-level field {field!r}")

    tasks = data.get("tasks")
    if not isinstance(tasks, list):
        return errors + ["top-level field 'tasks' must be a list"]

    required_fields = data.get("required_task_fields", [])
    if not isinstance(required_fields, list) or not required_fields:
        errors.append("top-level field 'required_task_fields' must be a non-empty list")
        required_fields = []

    if len(tasks) != 22:
        errors.append(f"expected 22 MLE-bench Lite tasks, found {len(tasks)}")

    seen_ids: set[str] = set()
    smoke_ids: list[str] = []

    for index, task in enumerate(tasks):
        prefix = f"tasks[{index}]"
        if not isinstance(task, dict):
            errors.append(f"{prefix} must be an object")
            continue

        task_id = task.get("id")
        if isinstance(task_id, str):
            if task_id in seen_ids:
                errors.append(f"{prefix}: duplicate task id {task_id!r}")
            seen_ids.add(task_id)

        for field in required_fields:
            if field not in task:
                errors.append(f"{prefix}: missing required field {field!r}")
            elif is_unresolved(task[field]):
                errors.append(f"{prefix}: unresolved required field {field!r}")

        if task.get("metric_direction") not in ALLOWED_DIRECTIONS:
            errors.append(f"{prefix}: metric_direction must be one of {sorted(ALLOWED_DIRECTIONS)}")

        if not isinstance(task.get("dataset_size_gb"), (int, float)):
            errors.append(f"{prefix}: dataset_size_gb must be numeric")

        if task.get("smoke") is True:
            smoke_ids.append(str(task_id))

    if len(smoke_ids) < 2:
        errors.append("smoke subset must contain at least 2 tasks")

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors = validate_manifest(args.manifest)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"validated {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
