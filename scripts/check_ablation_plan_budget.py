#!/usr/bin/env python3
"""Summarize planned ablation budget exposure from run manifests."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            records.append(value)
    return records


def collect_manifests(paths: list[Path], jsonl_paths: list[Path]) -> list[dict[str, Any]]:
    manifests = [load_json(path) for path in paths]
    for path in jsonl_paths:
        manifests.extend(iter_jsonl(path))
    if not manifests:
        raise ValueError("provide at least one --manifest or --manifest-jsonl")
    return manifests


def add_nested_cost(grouped: dict[str, dict[str, float]], group: str, cost: float) -> None:
    if group not in grouped:
        grouped[group] = {"count": 0, "max_cost_usd": 0.0}
    grouped[group]["count"] += 1
    grouped[group]["max_cost_usd"] += cost


def summarize_plan(manifests: list[dict[str, Any]], spend_cap_usd: float | None = None) -> dict[str, Any]:
    by_phase: dict[str, dict[str, float]] = {}
    by_system: dict[str, dict[str, float]] = {}
    by_phase_system: dict[str, dict[str, float]] = {}
    missing_budget = []
    total = 0.0

    for manifest in manifests:
        run_id = str(manifest.get("run_id", "<unknown>"))
        phase = str(manifest.get("phase", "unknown"))
        system = str(manifest.get("system", "unknown"))
        raw_cost = (manifest.get("budget") or {}).get("max_cost_usd")
        if raw_cost is None:
            missing_budget.append(run_id)
            cost = 0.0
        else:
            cost = float(raw_cost)
        total += cost
        add_nested_cost(by_phase, phase, cost)
        add_nested_cost(by_system, system, cost)
        add_nested_cost(by_phase_system, f"{phase}/{system}", cost)

    over_cap = spend_cap_usd is not None and total > spend_cap_usd
    return {
        "run_count": len(manifests),
        "total_max_cost_usd": round(total, 2),
        "spend_cap_usd": spend_cap_usd,
        "remaining_cap_usd": round(spend_cap_usd - total, 2) if spend_cap_usd is not None else None,
        "over_cap": bool(over_cap),
        "missing_budget_count": len(missing_budget),
        "missing_budget_run_ids": missing_budget[:50],
        "by_phase": {key: normalize_group(value) for key, value in sorted(by_phase.items())},
        "by_system": {key: normalize_group(value) for key, value in sorted(by_system.items())},
        "by_phase_system": {key: normalize_group(value) for key, value in sorted(by_phase_system.items())},
    }


def normalize_group(value: dict[str, float]) -> dict[str, int | float]:
    return {
        "count": int(value["count"]),
        "max_cost_usd": round(float(value["max_cost_usd"]), 2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path, default=[], help="Run manifest JSON; repeatable.")
    parser.add_argument("--manifest-jsonl", action="append", type=Path, default=[], help="Run manifests JSONL; repeatable.")
    parser.add_argument("--spend-cap-usd", type=float, help="Fail if planned max_cost_usd exceeds this cap.")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = summarize_plan(collect_manifests(args.manifest, args.manifest_jsonl), args.spend_cap_usd)
    encoded = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 2 if summary["over_cap"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"planned budget check failed: {exc}", file=sys.stderr)
        raise
