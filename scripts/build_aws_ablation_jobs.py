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
    return []


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


def ai_scientist_runtime_config_patch(
    *,
    budget: dict[str, Any] | None = None,
    runtime_controls: dict[str, Any] | None = None,
) -> str:
    budget = budget or {}
    runtime_controls = runtime_controls or {}
    max_nodes = budget.get("max_nodes")
    max_debug_attempts = budget.get("max_debug_attempts")
    wall_time_seconds = budget.get("wall_time_seconds")
    stage_budget = (
        ai_scientist_stage_budget(int(max_nodes))
        if runtime_controls.get("ai_scientist_budget_patch") and max_nodes is not None
        else {}
    )
    exec_timeout = (
        max(60, min(3600, int(wall_time_seconds)))
        if runtime_controls.get("ai_scientist_budget_patch") and wall_time_seconds is not None
        else None
    )
    debug_depth = max(0, min(3, int(max_debug_attempts))) if max_debug_attempts is not None else 1
    patch_parts = [
        "from pathlib import Path;"
        "p=Path('.context/external/AI-Scientist-v2-ablation/bfts_config.yaml');"
        "s=p.read_text();"
        "s=s.replace('model: anthropic.claude-3-5-sonnet-20241022-v2:0','model: gpt-4.1');"
        "s=s.replace('generate_report: True','generate_report: False');",
    ]
    if exec_timeout is not None:
        patch_parts.append(f"s=s.replace('  timeout: 3600','  timeout: {exec_timeout}');")
    # At screening budgets (>=8 nodes) keep 2 drafts so best-first selection
    # has multiple trees and genuinely differs from linear_stage.
    num_drafts = 2 if max_nodes is not None and int(max_nodes) >= 8 else 1
    if stage_budget:
        patch_parts.extend(
            [
                "s=s.replace('  num_workers: 4','  num_workers: 1');",
                "s=s.replace('    num_seeds: 3','    num_seeds: 1');",
                f"s=s.replace('    num_drafts: 3','    num_drafts: {num_drafts}');",
                f"s=s.replace('    max_debug_depth: 3','    max_debug_depth: {debug_depth}');",
                "s=s.replace('    max_tokens: 12000','    max_tokens: 6000');",
                "s=s.replace('    max_tokens: 8192','    max_tokens: 4000');",
                f"s=s.replace('    stage1_max_iters: 20','    stage1_max_iters: {stage_budget['stage1_max_iters']}');",
                f"s=s.replace('    stage2_max_iters: 12','    stage2_max_iters: {stage_budget['stage2_max_iters']}');",
                f"s=s.replace('    stage3_max_iters: 12','    stage3_max_iters: {stage_budget['stage3_max_iters']}');",
                f"s=s.replace('    stage4_max_iters: 18','    stage4_max_iters: {stage_budget['stage4_max_iters']}');",
                f"s=s.replace('  steps: 5','  steps: {stage_budget['steps']}');",
            ]
        )
    patch_parts.append(
        "p.write_text(s);"
        "lp=Path('.context/external/AI-Scientist-v2-ablation/launch_scientist_bfts.py');"
        "ls=lp.read_text();"
        "old='    aggregate_plots(base_folder=idea_dir, model=args.model_agg_plots)\\n\\n    shutil.rmtree(osp.join(idea_dir, \"experiment_results\"))';"
        "new='    if not args.skip_writeup:\\n        aggregate_plots(base_folder=idea_dir, model=args.model_agg_plots)\\n\\n    experiment_results_copy = osp.join(idea_dir, \"experiment_results\")\\n    if os.path.exists(experiment_results_copy):\\n        shutil.rmtree(experiment_results_copy)';"
        "ls=ls.replace(old,new);"
        "ls=ls.replace('    keywords = [\"python\", \"torch\", \"mp\", \"bfts\", \"experiment\"]','    keywords = []');"
        "lp.write_text(ls)"
    )
    patch_code = "".join(patch_parts)
    return f"python3 -c {shlex.quote(patch_code)}; "


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
    """Prepare task data when missing, then dispatch the ablation manifest."""
    sync_code = (
        "import sys,boto3;"
        "from pathlib import Path;"
        "src=Path(sys.argv[1]);"
        "dest=sys.argv[2].rstrip('/')+'/';"
        "bucket,key=dest.removeprefix('s3://').split('/',1);"
        "s3=boto3.client('s3');"
        "[s3.upload_file(str(p),bucket,key+str(p.relative_to(src))) for p in src.rglob('*') if p.is_file()]"
    )
    ai_experiment_copy_code = (
        "import shutil,sys;"
        "from pathlib import Path;"
        "src=Path(sys.argv[1]);"
        "dest=Path(sys.argv[2]);"
        "cutoff=float(sys.argv[3])-5;"
        "matched=[];"
        "\nif src.is_dir():\n"
        "    for p in src.iterdir():\n"
        "        if p.is_dir() and p.stat().st_mtime >= cutoff:\n"
        "            target=dest/p.name\n"
        "            shutil.copytree(p,target,dirs_exist_ok=True)\n"
        "            matched.append(str(target))\n"
        "print('\\n'.join(matched))"
    )
    ai_valid_grade_code = (
        "import json,sys;"
        "from pathlib import Path;"
        "p=Path(sys.argv[1]);"
        "ok=False;"
        "\nif p.is_file():\n"
        "    r=json.loads(p.read_text())\n"
        "    report=r.get('report') or {}\n"
        "    ok=bool(report.get('valid_submission'))\n"
        "print('1' if ok else '0')"
    )
    script = (
        "set -euo pipefail; "
        f"SYSTEM={shlex.quote(system)}; "
        f"TASK_ID={shlex.quote(task_id)}; "
        f"MANIFEST_REF={shlex.quote(manifest_ref)}; "
        f"PHASE={shlex.quote(phase)}; "
        f"RUN_ID={shlex.quote(run_id)}; "
        f"OUTPUT_DIR={shlex.quote(output_dir)}; "
        "RUN_START_TS=$(date +%s); "
        # Compute the artifact destination up-front so periodic checkpoints can
        # sync partial artifacts even if the job is killed at the wall timeout.
        'ARTIFACT_DEST=""; '
        'if [ -n "${ABLATION_ARTIFACT_ROOT:-}" ]; then '
        'ARTIFACT_DEST="${ABLATION_ARTIFACT_ROOT%/}/${PHASE}/${RUN_ID}/"; '
        "fi; "
        'test -n "${MLEBENCH_DATASET_DIR:-}" || '
        '{ echo "MLEBENCH_DATASET_DIR is required" >&2; exit 2; }; '
        'DESCRIPTION_PATH="${MLEBENCH_DATASET_DIR}/${TASK_ID}/prepared/public/description.md"; '
        'if [ ! -f "$DESCRIPTION_PATH" ]; then '
        'mlebench prepare -c "$TASK_ID" --data-dir "$MLEBENCH_DATASET_DIR"; '
        "fi; "
        + worker_patch_command(runtime_controls)
        + (
            ai_scientist_runtime_config_patch(
                budget=budget,
                runtime_controls=runtime_controls,
            )
            if system == "ai_scientist_v2"
            else ""
        )
        + "set +e; "
        # Background checkpoint loop: periodically sync partial artifacts so a
        # wall-timeout kill cannot lose everything before the final sync.
        'CHECKPOINT_PID=""; '
        'if [ -n "$ARTIFACT_DEST" ]; then '
        "( while true; do sleep 300; "
        f"python3 -c {shlex.quote(sync_code)} "
        '"$OUTPUT_DIR" "$ARTIFACT_DEST" >/dev/null 2>&1 || true; done ) & '
        "CHECKPOINT_PID=$!; "
        "fi; "
        'python3 scripts/run_ablation_manifest.py --manifest "$MANIFEST_REF"; '
        "RUN_STATUS=$?; "
        'if [ "$SYSTEM" = "ai_scientist_v2" ]; then '
        'AI_EXPERIMENTS_DIR=".context/external/AI-Scientist-v2-ablation/experiments"; '
        'AI_EXPERIMENTS_OUT="$OUTPUT_DIR/ai_scientist_experiments"; '
        f"python3 -c {shlex.quote(ai_experiment_copy_code)} "
        '"$AI_EXPERIMENTS_DIR" "$AI_EXPERIMENTS_OUT" "$RUN_START_TS"; '
        'AI_WORKSPACES_DIR=".context/external/AI-Scientist-v2-ablation/workspaces"; '
        'AI_WORKSPACES_OUT="$OUTPUT_DIR/ai_scientist_workspaces"; '
        f"python3 -c {shlex.quote(ai_experiment_copy_code)} "
        '"$AI_WORKSPACES_DIR" "$AI_WORKSPACES_OUT" "$RUN_START_TS"; '
        'python3 scripts/recover_ai_scientist_submission.py --output-dir "$OUTPUT_DIR" '
        '--task-id "$TASK_ID" --data-dir "$MLEBENCH_DATASET_DIR" --run-id "$RUN_ID" '
        '--timeout-seconds 900 --allow-fallback || true; '
        "fi; "
        'if [ -d "$OUTPUT_DIR" ]; then '
        'python3 scripts/grade_ablation_submissions.py --output-dir "$OUTPUT_DIR" '
        '--task-id "$TASK_ID" --system "$SYSTEM" --run-id "$RUN_ID" '
        '--data-dir "$MLEBENCH_DATASET_DIR" || true; '
        "fi; "
        'if [ "$SYSTEM" = "ai_scientist_v2" ] && [ -f "$OUTPUT_DIR/grader/grade_report.json" ]; then '
        f"VALID_GRADE=$(python3 -c {shlex.quote(ai_valid_grade_code)} "
        '"$OUTPUT_DIR/grader/grade_report.json"); '
        'if [ "$VALID_GRADE" = "1" ]; then RUN_STATUS=0; fi; '
        "fi; "
        'if [ -n "$CHECKPOINT_PID" ]; then kill "$CHECKPOINT_PID" >/dev/null 2>&1 || true; fi; '
        "SYNC_STATUS=0; "
        'if [ -n "$ARTIFACT_DEST" ] && [ -d "$OUTPUT_DIR" ]; then '
        f"python3 -c {shlex.quote(sync_code)} "
        '"$OUTPUT_DIR" "$ARTIFACT_DEST"; '
        "SYNC_STATUS=$?; "
        "fi; "
        'if [ "$RUN_STATUS" -ne 0 ]; then exit "$RUN_STATUS"; fi; '
        'exit "$SYNC_STATUS"'
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
            + ARTIFACT_SYNC_TIMEOUT_BUFFER_SECONDS
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
