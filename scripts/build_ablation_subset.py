#!/usr/bin/env python3
"""Clone selected ablation manifests into a smaller capped run subset."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / ".context/ablation/run_manifests"


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


def parse_system_variant(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("expected SYSTEM:VARIANT_ID")
    system, variant_id = value.split(":", 1)
    if not system or not variant_id:
        raise argparse.ArgumentTypeError("expected SYSTEM:VARIANT_ID")
    return system, variant_id


def selected(manifest: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.task and manifest.get("task_id") not in set(args.task):
        return False
    if args.seed and int(manifest.get("seed")) not in set(args.seed):
        return False
    if args.system_variant:
        identity = (str(manifest.get("system")), str(manifest.get("variant_id")))
        if identity not in set(args.system_variant):
            return False
    return True


def cheap_mlevolve_controls() -> dict[str, Any]:
    return {
        "agent.initial_drafts": 1,
        "agent.search.num_drafts": 1,
        "agent.search.parallel_search_num": 1,
        "agent.search.max_debug_depth": 1,
    }


def clone_manifest(manifest: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cloned = deepcopy(manifest)
    original_run_id = str(manifest["run_id"])
    new_run_id = f"{args.run_id_prefix}-{original_run_id}"
    cloned["run_id"] = new_run_id
    cloned["phase"] = args.phase
    cloned["status"] = "planned"

    budget = dict(cloned.get("budget") or {})
    if args.max_cost_usd is not None:
        budget["max_cost_usd"] = args.max_cost_usd
    if args.wall_time_seconds is not None:
        budget["wall_time_seconds"] = args.wall_time_seconds
    if args.max_nodes is not None:
        budget["max_nodes"] = args.max_nodes
    if args.max_input_tokens is not None:
        budget["max_input_tokens"] = args.max_input_tokens
    if args.max_output_tokens is not None:
        budget["max_output_tokens"] = args.max_output_tokens
    if args.max_debug_attempts is not None:
        budget["max_debug_attempts"] = args.max_debug_attempts
    budget["budget_policy_version"] = args.budget_policy_version
    budget["early_stop"] = {
        "min_nodes_before_check": 1,
        "catastrophic_valid_node_rate": 0.0,
    }
    cloned["budget"] = budget

    config_overrides = dict(cloned.get("config_overrides") or {})
    if args.mlevolve_cheap_controls and cloned.get("system") == "mlevolve":
        config_overrides.update(cheap_mlevolve_controls())
    cloned["config_overrides"] = config_overrides

    runtime_controls = dict(cloned.get("runtime_controls") or {})
    if args.ai_scientist_budget_patch and cloned.get("system") == "ai_scientist_v2":
        runtime_controls["ai_scientist_budget_patch"] = True
    if runtime_controls:
        cloned["runtime_controls"] = runtime_controls

    run_dir = args.output_root.parent / "runs" / new_run_id
    manifest_path = args.output_root / f"{new_run_id}.json"
    artifacts = dict(cloned.get("artifacts") or {})
    artifacts["output_dir"] = repo_relative_or_absolute(run_dir)
    artifacts["manifest_path"] = repo_relative_or_absolute(manifest_path)
    cloned["artifacts"] = artifacts

    notes = cloned.get("notes")
    suffix = f"Micro subset cloned from {original_run_id} with run_id_prefix={args.run_id_prefix}."
    cloned["notes"] = f"{notes} {suffix}".strip() if notes else suffix
    cloned["source_run_id"] = original_run_id
    return cloned


def repo_relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_manifest(manifest: dict[str, Any], force: bool) -> None:
    path = ROOT / manifest["artifacts"]["manifest_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", action="append", type=Path, required=True)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id-prefix", required=True)
    parser.add_argument("--phase", default="smoke")
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--system-variant", action="append", type=parse_system_variant, default=[])
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--max-cost-usd", type=float)
    parser.add_argument("--wall-time-seconds", type=int)
    parser.add_argument("--max-nodes", type=int)
    parser.add_argument("--max-input-tokens", type=int)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--max-debug-attempts", type=int)
    parser.add_argument("--budget-policy-version", default="ablation-micro-budget-v1")
    parser.add_argument("--mlevolve-cheap-controls", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ai-scientist-budget-patch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_root = args.output_root.resolve()

    source_manifests: list[dict[str, Any]] = []
    for path in args.source_jsonl:
        source_manifests.extend(iter_jsonl(path))

    manifests = [clone_manifest(manifest, args) for manifest in source_manifests if selected(manifest, args)]
    if args.max_runs is not None:
        manifests = manifests[: args.max_runs]
    if not manifests:
        raise ValueError("no manifests selected")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for manifest in manifests:
            write_manifest(manifest, args.force)
            handle.write(json.dumps(manifest, sort_keys=True) + "\n")

    total_cost = sum(float((manifest.get("budget") or {}).get("max_cost_usd", 0) or 0) for manifest in manifests)
    print(f"wrote {len(manifests)} manifests to {args.output_jsonl}; max_cost_usd={total_cost:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
