#!/usr/bin/env python3
"""Tests for collect_aws_batch_status.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COLLECT = ROOT / "scripts/collect_aws_batch_status.py"


def test_collect_from_describe_fixture() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        describe_path = tmp / "describe.json"
        output_jsonl = tmp / "status.jsonl"
        summary_json = tmp / "summary.json"
        describe_path.write_text(
            json.dumps(
                {
                    "jobs": [
                        {
                            "jobId": "job-1",
                            "jobName": "run-1",
                            "status": "SUCCEEDED",
                            "createdAt": 1,
                            "startedAt": 2,
                            "stoppedAt": 3,
                            "tags": {"phase": "smoke", "system": "mlevolve", "variant": "default_mcgs", "task": "spooky"},
                            "container": {"exitCode": 0, "logStreamName": "log-1"},
                        },
                        {
                            "jobId": "job-2",
                            "jobName": "run-2",
                            "status": "FAILED",
                            "statusReason": "Essential container exited",
                            "parameters": {
                                "phase": "smoke",
                                "system": "ai_scientist_v2",
                                "variant_id": "default",
                                "task_id": "spooky",
                            },
                            "container": {"exitCode": 1, "reason": "runtime error", "logStreamName": "log-2"},
                        },
                    ]
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        result = subprocess.run(
            [
                sys.executable,
                str(COLLECT),
                "--describe-json",
                str(describe_path),
                "--output-jsonl",
                str(output_jsonl),
                "--summary-json",
                str(summary_json),
            ],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert result.returncode == 0, result.stderr
        rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 2
        assert rows[0]["status"] == "SUCCEEDED"
        assert rows[1]["status"] == "FAILED"
        assert rows[1]["reason"] == "runtime error"

        summary = json.loads(summary_json.read_text(encoding="utf-8"))
        assert summary["job_count"] == 2
        assert summary["terminal_count"] == 2
        assert summary["all_terminal"] is True
        assert summary["status_counts"] == {"FAILED": 1, "SUCCEEDED": 1}
        assert summary["system_counts"] == {"ai_scientist_v2": 1, "mlevolve": 1}
        assert summary["failed_jobs"][0]["logStreamName"] == "log-2"


if __name__ == "__main__":
    test_collect_from_describe_fixture()
    print("collect_aws_batch_status tests passed")
