#!/usr/bin/env python3
"""Collect and summarize AWS Batch job status for ablation runs."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


TERMINAL_STATUSES = {"SUCCEEDED", "FAILED"}
PARAMETER_KEYS = {
    "variant": "variant_id",
    "task": "task_id",
}


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


def chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def describe_jobs(job_ids: list[str], aws_cli: str, region: str | None) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for batch in chunks(job_ids, 100):
        command = [aws_cli, "batch", "describe-jobs", "--jobs", *batch, "--output", "json"]
        if region:
            command.extend(["--region", region])
        completed = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "describe-jobs failed")
        payload = json.loads(completed.stdout)
        jobs.extend(payload.get("jobs", []))
    return jobs


def job_metadata(job: dict[str, Any], key: str) -> str:
    tags = job.get("tags") or {}
    if tags.get(key):
        return str(tags[key])
    parameters = job.get("parameters") or {}
    value = parameters.get(PARAMETER_KEYS.get(key, key), "")
    return str(value) if value is not None else ""


def normalize_job(job: dict[str, Any]) -> dict[str, Any]:
    container = job.get("container") or {}
    return {
        "jobId": job.get("jobId"),
        "jobName": job.get("jobName"),
        "status": job.get("status"),
        "statusReason": job.get("statusReason", ""),
        "createdAt": job.get("createdAt"),
        "startedAt": job.get("startedAt"),
        "stoppedAt": job.get("stoppedAt"),
        "exitCode": container.get("exitCode"),
        "reason": container.get("reason", ""),
        "logStreamName": container.get("logStreamName", ""),
        "phase": job_metadata(job, "phase"),
        "system": job_metadata(job, "system"),
        "variant": job_metadata(job, "variant"),
        "task": job_metadata(job, "task"),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(row.get("status", "UNKNOWN") for row in rows)
    system_counts = Counter(row.get("system", "unknown") for row in rows)
    terminal = sum(status_counts.get(status, 0) for status in TERMINAL_STATUSES)
    return {
        "job_count": len(rows),
        "terminal_count": terminal,
        "all_terminal": terminal == len(rows) if rows else False,
        "status_counts": dict(sorted(status_counts.items())),
        "system_counts": dict(sorted(system_counts.items())),
        "failed_jobs": [
            {
                "jobId": row.get("jobId"),
                "jobName": row.get("jobName"),
                "statusReason": row.get("statusReason"),
                "reason": row.get("reason"),
                "logStreamName": row.get("logStreamName"),
            }
            for row in rows
            if row.get("status") == "FAILED"
        ],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submitted-jsonl", type=Path, help="JSONL emitted by submit_aws_ablation_jobs.py.")
    parser.add_argument("--job-id", action="append", default=[], help="Explicit Batch job id; repeatable.")
    parser.add_argument("--describe-json", type=Path, help="Use saved aws batch describe-jobs JSON instead of calling AWS.")
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--aws-cli", default="aws")
    parser.add_argument("--region")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    job_ids = list(args.job_id)
    if args.submitted_jsonl:
        for record in iter_jsonl(args.submitted_jsonl):
            if record.get("jobId"):
                job_ids.append(str(record["jobId"]))

    if args.describe_json:
        payload = json.loads(args.describe_json.read_text(encoding="utf-8"))
        jobs = payload.get("jobs", [])
    else:
        if not job_ids:
            raise ValueError("provide --submitted-jsonl, --job-id, or --describe-json")
        jobs = describe_jobs(job_ids, aws_cli=args.aws_cli, region=args.region)

    rows = [normalize_job(job) for job in jobs]
    summary = summarize(rows)
    write_jsonl(args.output_jsonl, rows)

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
