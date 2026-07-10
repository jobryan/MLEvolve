#!/usr/bin/env python3
"""Tests for check_ablation_launch_readiness.py."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_ablation_launch_readiness.py"


def manifest(run_id: str = "run-1", cost: float = 50.0) -> dict:
    return {
        "run_id": run_id,
        "phase": "smoke",
        "system": "mlevolve",
        "budget": {"max_cost_usd": cost},
    }


def job_spec(job_name: str = "job-1", queue: str = "queue") -> dict:
    return {
        "jobName": job_name,
        "jobQueue": queue,
        "jobDefinition": "jobdef",
        "parameters": {
            "phase": "smoke",
            "system": "mlevolve",
            "variant_id": "default_mcgs",
            "task_id": "spooky-author-identification",
        },
        "containerOverrides": {
            "command": ["python3", "scripts/run_ablation_manifest.py"],
            "resourceRequirements": [{"type": "VCPU", "value": "8"}],
        },
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")


def run_guard(args: list[str]) -> subprocess.CompletedProcess[str]:
    # Drop OPENAI_API_KEY so the model-availability lookup never fires in tests.
    env = {key: value for key, value in os.environ.items() if key != "OPENAI_API_KEY"}
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )


def test_ready_with_cap_and_skipped_live_checks() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        manifests = tmp / "manifests.jsonl"
        jobs = tmp / "jobs.jsonl"
        write_jsonl(manifests, [manifest()])
        write_jsonl(jobs, [job_spec()])
        result = run_guard(
            [
                "--manifest-jsonl",
                str(manifests),
                "--jobs-jsonl",
                str(jobs),
                "--spend-cap-usd",
                "100",
                "--skip-kaggle",
                "--skip-active-jobs",
            ]
        )
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["ready"] is True


def test_missing_spend_cap_blocks() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        manifests = tmp / "manifests.jsonl"
        write_jsonl(manifests, [manifest()])
        result = run_guard(["--manifest-jsonl", str(manifests), "--skip-kaggle", "--skip-active-jobs"])
        summary = json.loads(result.stdout)
        assert result.returncode == 2
        assert "missing_spend_cap_usd" in summary["blockers"]


def test_placeholders_block() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        manifests = tmp / "manifests.jsonl"
        jobs = tmp / "jobs.jsonl"
        write_jsonl(manifests, [manifest()])
        write_jsonl(jobs, [job_spec(queue="TO_BE_REPLACED_QUEUE")])
        result = run_guard(
            [
                "--manifest-jsonl",
                str(manifests),
                "--jobs-jsonl",
                str(jobs),
                "--spend-cap-usd",
                "100",
                "--skip-kaggle",
                "--skip-active-jobs",
            ]
        )
        summary = json.loads(result.stdout)
        assert result.returncode == 2
        assert "job_specs_contain_placeholders" in summary["blockers"]


def routing_job_spec(variant_id: str, environment: list[dict] | None) -> dict:
    spec = job_spec(job_name=f"job-{variant_id}")
    spec["parameters"]["variant_id"] = variant_id
    if environment is not None:
        spec["containerOverrides"]["environment"] = environment
    return spec


def _run_model_tier_case(spec: dict) -> dict:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        manifests = tmp / "manifests.jsonl"
        jobs = tmp / "jobs.jsonl"
        write_jsonl(manifests, [manifest()])
        write_jsonl(jobs, [spec])
        result = run_guard(
            [
                "--manifest-jsonl",
                str(manifests),
                "--jobs-jsonl",
                str(jobs),
                "--spend-cap-usd",
                "100",
                "--skip-kaggle",
                "--skip-active-jobs",
            ]
        )
        return json.loads(result.stdout)


def test_model_tier_placebo_blocks() -> None:
    # all_strong routed to the anchor model = the tranche-1 placebo failure.
    summary = _run_model_tier_case(
        routing_job_spec(
            "all_strong",
            [
                {"name": "MLEVOLVE_STRONG_CODE_MODEL", "value": "gpt-4.1"},
                {"name": "MLEVOLVE_STRONG_FEEDBACK_MODEL", "value": "gpt-4.1"},
            ],
        )
    )
    assert "model_routing_tier_missing_or_placebo" in summary["blockers"]

    # Missing tier env entirely must also block.
    summary = _run_model_tier_case(routing_job_spec("all_strong", None))
    assert "model_routing_tier_missing_or_placebo" in summary["blockers"]

    # Cheap tiers routed to the anchor are placebos of the default too.
    summary = _run_model_tier_case(
        routing_job_spec(
            "all_cheap",
            [
                {"name": "MLEVOLVE_CHEAP_CODE_MODEL", "value": "gpt-4.1"},
                {"name": "MLEVOLVE_CHEAP_FEEDBACK_MODEL", "value": "gpt-4.1"},
            ],
        )
    )
    assert "model_routing_tier_missing_or_placebo" in summary["blockers"]

    summary = _run_model_tier_case(
        routing_job_spec(
            "cheap_code_strong_feedback",
            [
                {"name": "MLEVOLVE_CHEAP_CODE_MODEL", "value": "gpt-4.1"},
                {"name": "MLEVOLVE_STRONG_FEEDBACK_MODEL", "value": "gpt-5.5"},
            ],
        )
    )
    assert "model_routing_tier_missing_or_placebo" in summary["blockers"]

    # Missing tier env on the new arms must also block.
    summary = _run_model_tier_case(routing_job_spec("all_cheap", None))
    assert "model_routing_tier_missing_or_placebo" in summary["blockers"]


def test_model_tier_real_tier_passes() -> None:
    summary = _run_model_tier_case(
        routing_job_spec(
            "strong_code_cheap_feedback",
            [
                {"name": "MLEVOLVE_STRONG_CODE_MODEL", "value": "gpt-5.5"},
                {"name": "MLEVOLVE_CHEAP_FEEDBACK_MODEL", "value": "gpt-4.1-mini"},
            ],
        )
    )
    assert "model_routing_tier_missing_or_placebo" not in summary["blockers"]
    assert summary["model_tiers"]["routing_job_count"] == 1
    assert summary["ready"] is True

    summary = _run_model_tier_case(
        routing_job_spec(
            "cheap_code_strong_feedback",
            [
                {"name": "MLEVOLVE_CHEAP_CODE_MODEL", "value": "gpt-4.1-mini"},
                {"name": "MLEVOLVE_STRONG_FEEDBACK_MODEL", "value": "gpt-5.5"},
            ],
        )
    )
    assert "model_routing_tier_missing_or_placebo" not in summary["blockers"]
    assert summary["ready"] is True

    summary = _run_model_tier_case(
        routing_job_spec(
            "all_cheap",
            [
                {"name": "MLEVOLVE_CHEAP_CODE_MODEL", "value": "gpt-4.1-mini"},
                {"name": "MLEVOLVE_CHEAP_FEEDBACK_MODEL", "value": "gpt-4.1-mini"},
            ],
        )
    )
    assert "model_routing_tier_missing_or_placebo" not in summary["blockers"]
    assert summary["ready"] is True


if __name__ == "__main__":
    test_ready_with_cap_and_skipped_live_checks()
    test_missing_spend_cap_blocks()
    test_placeholders_block()
    test_model_tier_placebo_blocks()
    test_model_tier_real_tier_passes()
    print("check_ablation_launch_readiness tests passed")
