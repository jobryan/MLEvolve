#!/usr/bin/env python3
"""Check whether an ablation run exceeded its manifest budget."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


LIMIT_FIELDS = {
    "wall_time_seconds": ("actual", "wall_time_seconds"),
    "max_nodes": ("actual", "nodes"),
    "max_cost_usd": ("actual", "cost_usd"),
    "max_input_tokens": ("token_usage", "input_tokens"),
    "max_output_tokens": ("token_usage", "output_tokens"),
    "max_debug_attempts": ("actual", "debug_attempts"),
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def nested_get(data: dict[str, Any], path: tuple[str, str]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def check_budget(manifest: dict[str, Any], run_record: dict[str, Any]) -> dict[str, Any]:
    budget = manifest.get("budget", {})
    overages = []

    for budget_field, actual_path in LIMIT_FIELDS.items():
        limit = budget.get(budget_field)
        actual = nested_get(run_record, actual_path)
        if limit is None or actual is None:
            continue
        if actual > limit:
            overages.append(
                {
                    "budget_field": budget_field,
                    "limit": limit,
                    "actual_field": ".".join(actual_path),
                    "actual": actual,
                }
            )

    status = "budget_exceeded" if overages else run_record.get("status", "success")
    return {
        "run_id": manifest.get("run_id"),
        "status": status,
        "over_budget": bool(overages),
        "overages": overages,
        "early_stop": budget.get("early_stop", {}),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-record", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = check_budget(load_json(args.manifest), load_json(args.run_record))
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0 if not result["over_budget"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"budget check failed: {exc}", file=sys.stderr)
        raise
