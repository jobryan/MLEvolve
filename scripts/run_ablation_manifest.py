#!/usr/bin/env python3
"""Run or dry-run a single ablation manifest on an execution worker."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AIS_ROOT = ROOT / ".context/external/AI-Scientist-v2-ablation"
AIS_COMMAND_BUILDER = AIS_ROOT / "ablation_adapter/build_variant_command.py"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def load_manifest(ref: str) -> tuple[dict[str, Any], Path | None]:
    if not ref.startswith("s3://"):
        return load_json(Path(ref).resolve()), None

    # Import lazily so dry-run/local command construction does not require boto3.
    import boto3  # type: ignore[import-not-found]

    without_scheme = ref.removeprefix("s3://")
    bucket, _, key = without_scheme.partition("/")
    if not bucket or not key:
        raise ValueError(f"invalid S3 manifest URI: {ref}")
    temp_dir = Path(tempfile.mkdtemp(prefix="ablation-manifest-"))
    local_path = temp_dir / Path(key).name
    boto3.client("s3").download_file(bucket, key, str(local_path))
    return load_json(local_path), local_path


def bool_value(value: bool) -> str:
    return "true" if value else "false"


def cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return bool_value(value)
    if value is None:
        return "null"
    return str(value)


def mlevolve_command(manifest: dict[str, Any], python_bin: str) -> dict[str, Any]:
    task_id = manifest["task_id"]
    run_id = manifest["run_id"]
    output_dir = ROOT / manifest["artifacts"]["output_dir"]
    dataset_root = os.environ.get("MLEBENCH_DATASET_DIR", "${MLEBENCH_DATASET_DIR}")
    data_root = Path(dataset_root) / task_id / "prepared/public"
    desc_file = data_root / "description.md"
    budget = manifest.get("budget", {})

    overrides = {
        "exp_id": task_id,
        "dataset_dir": dataset_root,
        "data_dir": str(data_root),
        "desc_file": str(desc_file),
        "exp_name": run_id,
        "log_dir": str(output_dir),
        "workspace_dir": str(output_dir),
        "agent.seed": manifest["seed"],
        "agent.steps": budget.get("max_nodes"),
        "agent.time_limit": budget.get("wall_time_seconds"),
        "exec.timeout": budget.get("wall_time_seconds"),
        "ablation.enabled": True,
        "ablation.run_id": run_id,
        "ablation.variant_id": manifest["variant_id"],
        "ablation.benchmark_track": manifest["benchmark_track"],
        "ablation.task_id": task_id,
        "ablation.phase": manifest["phase"],
        "ablation.task_manifest_version": manifest.get("task_manifest_version", ""),
        "ablation.variant_registry_version": manifest.get("variant_registry_version", ""),
        # Keep secrets out of manifests/job specs. OmegaConf resolves these on
        # the worker from injected environment variables. The default ablation
        # runtime uses OpenAI-compatible models; Gemini can still be selected by
        # explicit model/API-key overrides outside the default plan.
        "agent.code.model": "${oc.env:MLEVOLVE_CODE_MODEL,${oc.env:MLEVOLVE_STRONG_CODE_MODEL,gpt-4.1}}",
        "agent.feedback.model": "${oc.env:MLEVOLVE_FEEDBACK_MODEL,${oc.env:MLEVOLVE_STRONG_FEEDBACK_MODEL,gpt-4.1}}",
        "agent.code.api_key": "${oc.env:MLEVOLVE_CODE_API_KEY,${oc.env:OPENAI_API_KEY,}}",
        "agent.feedback.api_key": "${oc.env:MLEVOLVE_FEEDBACK_API_KEY,${oc.env:OPENAI_API_KEY,}}",
        "agent.code.base_url": "${oc.env:MLEVOLVE_CODE_BASE_URL,}",
        "agent.feedback.base_url": "${oc.env:MLEVOLVE_FEEDBACK_BASE_URL,}",
    }
    overrides.update(manifest.get("config_overrides") or {})

    command = [python_bin, "run.py"]
    for key, value in overrides.items():
        if value is None:
            continue
        command.append(f"{key}={cli_value(value)}")

    return {
        "cwd": str(ROOT),
        "command": command,
        "command_string": " ".join(shlex.quote(part) for part in command),
        "env": {},
        "output_dir": str(output_dir),
    }


def ai_scientist_idea_path(manifest: dict[str, Any]) -> Path:
    task_id = manifest["task_id"]
    seed = manifest["seed"]
    return AIS_ROOT / f"ablation_adapter/generated_ideas/mle_bench_{task_id}_seed_{seed}_implementation.json"


def ai_scientist_command(manifest: dict[str, Any], python_bin: str) -> dict[str, Any]:
    config_path = manifest.get("variant", {}).get("config_path")
    if not config_path:
        raise ValueError(f"manifest {manifest['run_id']} has no variant.config_path")
    idea_path = ai_scientist_idea_path(manifest)
    command = [
        python_bin,
        str(AIS_COMMAND_BUILDER.relative_to(ROOT)),
        "--variant",
        config_path,
        "--ideas",
        str(idea_path),
        "--attempt-id",
        str(manifest["seed"]),
        "--python",
        python_bin,
    ]
    task = manifest.get("task") or {}
    sample_submission = task.get("sample_submission_path")
    dataset_root = os.environ.get("MLEBENCH_DATASET_DIR")
    if sample_submission:
        sample_submission_path = Path(sample_submission)
        if not sample_submission_path.is_absolute() and dataset_root:
            sample_submission_path = Path(dataset_root) / sample_submission_path
        sample_submission_value = str(sample_submission_path)
    else:
        sample_submission_value = ""

    return {
        "cwd": str(ROOT),
        "command": command,
        "command_string": " ".join(shlex.quote(part) for part in command),
        "env": {
            "ABLATION_RUN_ID": manifest["run_id"],
            "ABLATION_OUTPUT_DIR": manifest["artifacts"]["output_dir"],
            "ABLATION_TASK_ID": manifest["task_id"],
            "ABLATION_SHARED_BENCHMARK": "1",
            "ABLATION_AI_SCIENTIST_IDEA_PATH": str(idea_path),
            "ABLATION_SAMPLE_SUBMISSION_PATH": sample_submission_value,
            "ABLATION_METRIC_NAME": task.get("metric_name") or "",
            "ABLATION_METRIC_DIRECTION": task.get("metric_direction") or "",
            "ABLATION_SKIP_AI_PLOTS": "1",
        },
        "output_dir": str(ROOT / manifest["artifacts"]["output_dir"]),
        "idea_path": str(idea_path),
    }


def build_command_spec(manifest: dict[str, Any], python_bin: str) -> dict[str, Any]:
    system = manifest["system"]
    if system == "mlevolve":
        runner = mlevolve_command(manifest, python_bin)
    elif system == "ai_scientist_v2":
        runner = ai_scientist_command(manifest, python_bin)
    else:
        raise ValueError(f"unsupported system {system!r}")

    return {
        "schema_version": "1.0",
        "run_id": manifest["run_id"],
        "system": system,
        "variant_id": manifest["variant_id"],
        "task_id": manifest["task_id"],
        "phase": manifest["phase"],
        "seed": manifest["seed"],
        "runner": runner,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Local path or s3:// URI for a manifest JSON file.")
    parser.add_argument("--python", default=os.environ.get("PYTHON", "python3"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest, _ = load_manifest(args.manifest)
    command_spec = build_command_spec(manifest, args.python)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(command_spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.dry_run:
        print(json.dumps(command_spec, indent=2, sort_keys=True))
        return 0

    output_dir = Path(command_spec["runner"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    copied_manifest = output_dir / "manifest.json"
    copied_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env.update(command_spec["runner"].get("env", {}))
    completed = subprocess.run(
        command_spec["runner"]["command"],
        cwd=command_spec["runner"]["cwd"],
        env=env,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
