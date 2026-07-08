#!/usr/bin/env python3
"""Tests for submit_aws_ablation_jobs.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "scripts/submit_aws_ablation_jobs.py"


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def job_spec(job_name: str, queue: str = "queue", definition: str = "jobdef") -> dict:
    return {
        "jobName": job_name,
        "jobQueue": queue,
        "jobDefinition": definition,
        "parameters": {
            "phase": "smoke",
            "system": "mlevolve",
            "variant_id": "default_mcgs",
            "task_id": "spooky-author-identification",
        },
        "containerOverrides": {
            "command": ["python3", "scripts/run_ablation_manifest.py", "--manifest", "s3://bucket/manifest.json"],
            "resourceRequirements": [{"type": "VCPU", "value": "8"}, {"type": "MEMORY", "value": "32768"}],
        },
        "tags": {
            "phase": "smoke",
            "system": "mlevolve",
            "variant": "default_mcgs",
            "task": "spooky-author-identification",
        },
    }


def run_submit(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SUBMIT), *args],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_dry_run_summary() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        jobs_path = tmp / "jobs.jsonl"
        write_jsonl(jobs_path, [job_spec("job-1"), job_spec("job-2")])
        result = run_submit(["--jobs-jsonl", str(jobs_path), "--dry-run"])
        assert result.returncode == 0, result.stderr
        summary = json.loads(result.stdout)
        assert summary["job_count"] == 2
        assert summary["placeholder_count"] == 0
        assert summary["ready_for_submit"] is True
        assert summary["system_counts"] == {"mlevolve": 2}


def test_dry_run_summary_without_tags() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        jobs_path = tmp / "jobs.jsonl"
        spec = job_spec("job-1")
        spec.pop("tags")
        write_jsonl(jobs_path, [spec])
        result = run_submit(["--jobs-jsonl", str(jobs_path), "--dry-run"])
        assert result.returncode == 0, result.stderr
        summary = json.loads(result.stdout)
        assert summary["phase_counts"] == {"smoke": 1}
        assert summary["system_counts"] == {"mlevolve": 1}


def test_refuses_placeholders_without_explicit_dry_run_allowance() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        jobs_path = tmp / "jobs.jsonl"
        write_jsonl(jobs_path, [job_spec("job-1", queue="TO_BE_REPLACED_QUEUE")])

        strict_dry_run = run_submit(["--jobs-jsonl", str(jobs_path), "--dry-run"])
        assert strict_dry_run.returncode == 1
        assert "placeholder" in strict_dry_run.stderr

        allowed_dry_run = run_submit(["--jobs-jsonl", str(jobs_path), "--dry-run", "--allow-placeholders"])
        assert allowed_dry_run.returncode == 0, allowed_dry_run.stderr
        summary = json.loads(allowed_dry_run.stdout)
        assert summary["placeholder_count"] == 1
        assert summary["ready_for_submit"] is False

        real_submit = run_submit(["--jobs-jsonl", str(jobs_path), "--aws-cli", "false"])
        assert real_submit.returncode == 1
        assert "placeholder" in real_submit.stderr


if __name__ == "__main__":
    test_dry_run_summary()
    test_dry_run_summary_without_tags()
    test_refuses_placeholders_without_explicit_dry_run_allowance()
    print("submit_aws_ablation_jobs tests passed")
