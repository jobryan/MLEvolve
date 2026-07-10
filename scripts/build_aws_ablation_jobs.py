#!/usr/bin/env python3
"""Render AWS Batch job specs from ablation run manifests."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV = ROOT / "configs/ablations/aws_worker_environment.json"
ARTIFACT_SYNC_TIMEOUT_BUFFER_SECONDS = 300

# Model tiers for the model-routing block (matrix D). The anchor model is
# gpt-4.1 (run_ablation_manifest.py env-resolver fallback); the strong tier
# must differ from it or all_strong degenerates into a default replicate.
DEFAULT_STRONG_MODEL = "gpt-5.5"
DEFAULT_CHEAP_MODEL = "gpt-4.1-mini"
STRONG_MODEL_BUILD_ENV = "ABLATION_STRONG_MODEL"
CHEAP_MODEL_BUILD_ENV = "ABLATION_CHEAP_MODEL"


def model_tier_environment(variant_id: str) -> list[dict[str, str]]:
    """Tier env vars for model-routing jobs only; anchors must never inherit them."""
    strong = os.environ.get(STRONG_MODEL_BUILD_ENV, DEFAULT_STRONG_MODEL)
    cheap = os.environ.get(CHEAP_MODEL_BUILD_ENV, DEFAULT_CHEAP_MODEL)
    if variant_id == "all_strong":
        return [
            {"name": "MLEVOLVE_STRONG_CODE_MODEL", "value": strong},
            {"name": "MLEVOLVE_STRONG_FEEDBACK_MODEL", "value": strong},
        ]
    if variant_id == "strong_code_cheap_feedback":
        return [
            {"name": "MLEVOLVE_STRONG_CODE_MODEL", "value": strong},
            {"name": "MLEVOLVE_CHEAP_FEEDBACK_MODEL", "value": cheap},
        ]
    if variant_id == "cheap_code_strong_feedback":
        return [
            {"name": "MLEVOLVE_CHEAP_CODE_MODEL", "value": cheap},
            {"name": "MLEVOLVE_STRONG_FEEDBACK_MODEL", "value": strong},
        ]
    if variant_id == "all_cheap":
        return [
            {"name": "MLEVOLVE_CHEAP_CODE_MODEL", "value": cheap},
            {"name": "MLEVOLVE_CHEAP_FEEDBACK_MODEL", "value": cheap},
        ]
    return []


def artifact_sync_timeout_buffer_seconds(runtime_controls: dict[str, Any] | None = None) -> int:
    """Return post-agent Batch timeout buffer for grading and final artifact sync."""
    runtime_controls = runtime_controls or {}
    value = runtime_controls.get("batch_timeout_buffer_seconds", ARTIFACT_SYNC_TIMEOUT_BUFFER_SECONDS)
    try:
        seconds = int(value)
    except (TypeError, ValueError):
        seconds = ARTIFACT_SYNC_TIMEOUT_BUFFER_SECONDS
    return max(ARTIFACT_SYNC_TIMEOUT_BUFFER_SECONDS, seconds)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
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
        raise ValueError("no manifests provided")
    return manifests


def batch_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", value)
    return cleaned[:128].strip("-") or "ablation-job"


def resource_value(mapping: dict[str, Any], resource_class: str, default: int) -> str:
    return str(mapping.get(resource_class, default))


def ai_scientist_stage_budget(max_nodes: int | None) -> dict[str, int]:
    if max_nodes is None:
        return {}
    nodes = max(1, int(max_nodes))
    # In very small pilots, the useful question is whether AI Scientist can
    # produce any working MLE-bench implementation. Allocate the first few
    # iterations to the initial implementation stage before later refinements.
    # At screening budgets (>=8 nodes) stage 1 gets half the nodes so stage 4
    # cannot swallow the majority of the tree.
    stage1 = max(3, nodes // 2) if nodes >= 8 else min(nodes, 3)
    remaining = max(0, nodes - stage1)
    stage2 = min(1, remaining)
    remaining -= stage2
    stage3 = min(1, remaining)
    remaining -= stage3
    stage4 = max(0, remaining)
    return {
        "stage1_max_iters": stage1,
        "stage2_max_iters": stage2,
        "stage3_max_iters": stage3,
        "stage4_max_iters": stage4,
        "steps": nodes,
    }


def ai_scientist_budget_environment(
    budget: dict[str, Any] | None = None,
    runtime_controls: dict[str, Any] | None = None,
) -> dict[str, str]:
    """T3_AIS_* env consumed by scripts/worker_ais_budget_patch.py (via worker_run.sh).

    The bfts config patch itself lives in scripts/worker_ais_budget_patch.py
    (shipped in the worker patch tarball); this only computes its parameters so
    the containerOverrides command stays short.
    """
    budget = budget or {}
    runtime_controls = runtime_controls or {}
    if not runtime_controls.get("ai_scientist_budget_patch"):
        return {}

    env: dict[str, str] = {"T3_AIS_BUDGET_PATCH": "1"}

    wall_time_seconds = budget.get("wall_time_seconds")
    # Optional explicit per-node exec timeout override (runtime_controls.ais_exec_timeout);
    # the effective value is min(override, wall) clamped to [60, 3600].
    ais_exec_timeout = runtime_controls.get("ais_exec_timeout")
    candidates = [int(value) for value in (wall_time_seconds, ais_exec_timeout) if value is not None]
    if candidates:
        env["T3_AIS_EXEC_TIMEOUT"] = str(max(60, min(3600, min(candidates))))

    max_nodes = budget.get("max_nodes")
    if max_nodes is not None:
        stage_budget = ai_scientist_stage_budget(int(max_nodes))
        env["T3_AIS_STAGE1"] = str(stage_budget["stage1_max_iters"])
        env["T3_AIS_STAGE2"] = str(stage_budget["stage2_max_iters"])
        env["T3_AIS_STAGE3"] = str(stage_budget["stage3_max_iters"])
        env["T3_AIS_STAGE4"] = str(stage_budget["stage4_max_iters"])
        env["T3_AIS_STEPS"] = str(stage_budget["steps"])
        # At screening budgets (>=8 nodes) keep 2 drafts so best-first selection
        # has multiple trees and genuinely differs from linear_stage.
        env["T3_AIS_NUM_DRAFTS"] = "2" if int(max_nodes) >= 8 else "1"

    max_debug_attempts = budget.get("max_debug_attempts")
    env["T3_AIS_DEBUG_DEPTH"] = (
        str(max(0, min(3, int(max_debug_attempts)))) if max_debug_attempts is not None else "1"
    )
    return env


def worker_patch_command(runtime_controls: dict[str, Any] | None = None) -> str:
    runtime_controls = runtime_controls or {}
    patch_uri = runtime_controls.get("worker_patch_uri")
    if not patch_uri:
        return ""

    patch_code = (
        "import boto3,sys,tarfile,tempfile;"
        "from pathlib import Path;"
        "uri=sys.argv[1];dest=Path(sys.argv[2]).resolve();"
        "bucket,key=uri.removeprefix('s3://').split('/',1);"
        "tmp=Path(tempfile.mkdtemp(prefix='ablation-worker-patch-'))/'patch.tar.gz';"
        "boto3.client('s3').download_file(bucket,key,str(tmp));"
        "tar=tarfile.open(tmp,'r:gz');"
        "members=tar.getmembers();"
        "\nfor member in members:\n"
        "    target=(dest/member.name).resolve()\n"
        "    if dest not in target.parents and target != dest:\n"
        "        raise RuntimeError(f'unsafe patch path: {member.name}')\n"
        "tar.extractall(dest,members=members);"
        "tar.close()"
    )
    return f"python3 -c {shlex.quote(patch_code)} {shlex.quote(str(patch_uri))} .; "


def worker_command(
    system: str,
    task_id: str,
    manifest_ref: str,
    phase: str,
    run_id: str,
    output_dir: str,
    budget: dict[str, Any] | None = None,
    runtime_controls: dict[str, Any] | None = None,
) -> list[str]:
    """Short bootstrap command: export parameters, download/extract the worker
    patch, then hand off to scripts/worker_run.sh (shipped in the patch).

    All run logic (mlebench prepare, checkpoint sync, AIS budget patch, grading,
    exit-code semantics) lives in worker_run.sh so the AWS Batch
    containerOverrides JSON stays below the 8192-character limit.
    """
    exports: dict[str, str] = {
        "SYSTEM": system,
        "TASK_ID": task_id,
        "MANIFEST_REF": manifest_ref,
        "PHASE": phase,
        "RUN_ID": run_id,
        "OUTPUT_DIR": output_dir,
    }
    if system == "ai_scientist_v2":
        exports.update(ai_scientist_budget_environment(budget, runtime_controls))
    export_clause = "export " + " ".join(
        f"{name}={shlex.quote(str(value))}" for name, value in exports.items()
    )
    script = (
        "set -euo pipefail; "
        f"{export_clause}; "
        + worker_patch_command(runtime_controls)
        + "bash scripts/worker_run.sh"
    )
    return ["bash", "-lc", script]


def build_job_spec(
    manifest: dict[str, Any],
    env_config: dict[str, Any],
    job_queue: str,
    job_definition: str,
    manifest_base_uri: str | None,
    include_tags: bool = False,
) -> dict[str, Any]:
    aws_defaults = env_config["aws_batch_defaults"]
    resource_class = manifest.get("resource_policy", {}).get("resource_class", "small_cpu")
    manifest_path = manifest.get("artifacts", {}).get("manifest_path")
    if not manifest_path:
        raise ValueError(f"manifest {manifest.get('run_id')} missing artifacts.manifest_path")

    if manifest_base_uri:
        manifest_ref = f"{manifest_base_uri.rstrip('/')}/{Path(manifest_path).name}"
    else:
        manifest_ref = manifest_path

    resource_requirements = [
        {
            "type": "VCPU",
            "value": resource_value(aws_defaults["vcpus_by_resource_class"], resource_class, 8),
        },
        {
            "type": "MEMORY",
            "value": resource_value(aws_defaults["memory_mb_by_resource_class"], resource_class, 32768),
        },
    ]
    gpus = int(aws_defaults["gpus_by_resource_class"].get(resource_class, 0))
    if gpus > 0:
        resource_requirements.append({"type": "GPU", "value": str(gpus)})

    spec = {
        "jobName": batch_name(manifest["run_id"]),
        "jobQueue": job_queue,
        "jobDefinition": job_definition,
        "parameters": {
            "run_id": manifest["run_id"],
            "system": manifest["system"],
            "variant_id": manifest["variant_id"],
            "task_id": manifest["task_id"],
            "phase": manifest["phase"],
            "seed": str(manifest["seed"]),
        },
        "containerOverrides": {
            "command": worker_command(
                manifest["system"],
                manifest["task_id"],
                manifest_ref,
                manifest["phase"],
                manifest["run_id"],
                manifest["artifacts"]["output_dir"],
                manifest.get("budget") or {},
                manifest.get("runtime_controls") or {},
            ),
            "environment": [
                {"name": "ABLATION_RUN_ID", "value": manifest["run_id"]},
                {"name": "ABLATION_SYSTEM", "value": manifest["system"]},
                {"name": "ABLATION_VARIANT_ID", "value": manifest["variant_id"]},
                {"name": "ABLATION_TASK_ID", "value": manifest["task_id"]},
                {"name": "ABLATION_PHASE", "value": manifest["phase"]},
            ]
            + (
                [
                    {"name": "MLEVOLVE_NO_GPU", "value": "1"},
                    {"name": "MLEVOLVE_OFFLINE", "value": "1"},
                    {"name": "CUDA_VISIBLE_DEVICES", "value": "-1"},
                ]
                if gpus == 0
                else []
            )
            + model_tier_environment(manifest["variant_id"]),
            "resourceRequirements": resource_requirements,
        },
        "timeout": {
            "attemptDurationSeconds": int(manifest.get("budget", {}).get("wall_time_seconds", 14400))
            + artifact_sync_timeout_buffer_seconds(manifest.get("runtime_controls") or {})
        },
    }
    if include_tags:
        spec["tags"] = {
            "project": "mlevolve-ai-scientist-v2-ablation",
            "phase": manifest["phase"],
            "system": manifest["system"],
            "variant": manifest["variant_id"],
            "task": manifest["task_id"],
        }

    # AWS Batch rejects jobs whose container overrides exceed 8192 characters
    # ("Container Overrides length must be at most 8192"); fail the build with
    # headroom instead of failing at submit/runtime.
    overrides_length = len(json.dumps(spec["containerOverrides"]))
    if overrides_length > 8000:
        raise ValueError(
            f"containerOverrides JSON is {overrides_length} chars (> 8000) for "
            f"run_id={manifest['run_id']}; AWS Batch rejects overrides above 8192 chars"
        )
    return spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path, default=[], help="Manifest JSON file; repeatable.")
    parser.add_argument("--manifest-jsonl", action="append", type=Path, default=[], help="JSONL of manifest records.")
    parser.add_argument("--environment", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--job-queue", help="AWS Batch job queue. Defaults to environment config.")
    parser.add_argument("--job-definition", help="AWS Batch job definition. Defaults to environment config.")
    parser.add_argument("--manifest-base-uri", help="Optional S3/base URI where manifest files are available to workers.")
    parser.add_argument("--include-tags", action="store_true", help="Include Batch tags; requires batch:TagResource.")
    parser.add_argument("--output-jsonl", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env_config = load_json(args.environment)
    aws_defaults = env_config["aws_batch_defaults"]
    job_queue = args.job_queue or aws_defaults["job_queue"]
    job_definition = args.job_definition or aws_defaults["job_definition"]
    if job_queue == "TO_BE_CONFIGURED" or job_definition == "TO_BE_CONFIGURED":
        raise ValueError("--job-queue and --job-definition are required until AWS defaults are configured")

    manifests = collect_manifests(args.manifest, args.manifest_jsonl)
    specs = [
        build_job_spec(
            manifest,
            env_config,
            job_queue=job_queue,
            job_definition=job_definition,
            manifest_base_uri=args.manifest_base_uri,
            include_tags=args.include_tags,
        )
        for manifest in manifests
    ]

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for spec in specs:
            handle.write(json.dumps(spec, sort_keys=True) + "\n")
    print(f"wrote {len(specs)} AWS Batch job specs to {args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
