#!/usr/bin/env python3
"""Submit rendered AWS Batch ablation job specs, or validate them in dry-run mode."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


PLACEHOLDER_PREFIXES = (
    "TO_BE_REPLACED",
    "TO_BE_CONFIGURED",
)
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


def has_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        return any(token in value for token in PLACEHOLDER_PREFIXES)
    if isinstance(value, dict):
        return any(has_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(has_placeholder(item) for item in value)
    return False


def spec_metadata(spec: dict[str, Any], key: str, default: str | None = None) -> str | None:
    tags = spec.get("tags") or {}
    if tags.get(key):
        return tags[key]
    parameters = spec.get("parameters") or {}
    return parameters.get(PARAMETER_KEYS.get(key, key), default)


def job_summary(specs: list[dict[str, Any]]) -> dict[str, Any]:
    by_phase = Counter(spec_metadata(spec, "phase", "unknown") for spec in specs)
    by_system = Counter(spec_metadata(spec, "system", "unknown") for spec in specs)
    placeholders = [spec.get("jobName", "<unknown>") for spec in specs if has_placeholder(spec)]
    return {
        "job_count": len(specs),
        "phase_counts": dict(sorted(by_phase.items())),
        "system_counts": dict(sorted(by_system.items())),
        "placeholder_count": len(placeholders),
        "placeholder_jobs": placeholders[:20],
        "ready_for_submit": len(placeholders) == 0,
    }


def submit_spec(spec: dict[str, Any], aws_cli: str, region: str | None) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
        json.dump(spec, handle, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)

    command = [
        aws_cli,
        "batch",
        "submit-job",
        "--cli-input-json",
        f"file://{temp_path}",
        "--output",
        "json",
    ]
    if region:
        command.extend(["--region", region])

    completed = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    temp_path.unlink(missing_ok=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"submit-job failed for {spec.get('jobName')}")

    payload = json.loads(completed.stdout)
    return {
        "jobName": payload.get("jobName"),
        "jobId": payload.get("jobId"),
        "source_jobName": spec.get("jobName"),
        "phase": spec_metadata(spec, "phase"),
        "system": spec_metadata(spec, "system"),
        "variant": spec_metadata(spec, "variant"),
        "task": spec_metadata(spec, "task"),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", type=Path, help="Where to write submitted job IDs.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-jobs", type=int, help="Limit number of jobs submitted or dry-run counted.")
    parser.add_argument("--aws-cli", default="aws")
    parser.add_argument("--region")
    parser.add_argument("--allow-placeholders", action="store_true", help="Only allowed with --dry-run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specs = iter_jsonl(args.jobs_jsonl)
    if args.max_jobs is not None:
        specs = specs[: args.max_jobs]

    summary = job_summary(specs)
    print(json.dumps(summary, indent=2, sort_keys=True))

    if summary["placeholder_count"] and not (args.dry_run and args.allow_placeholders):
        print("job specs contain placeholder values; refusing to submit", file=sys.stderr)
        return 1

    if args.dry_run:
        return 0

    submitted: list[dict[str, Any]] = []
    for spec in specs:
        submitted.append(submit_spec(spec, aws_cli=args.aws_cli, region=args.region))

    output_jsonl = args.output_jsonl or args.jobs_jsonl.with_suffix(".submitted.jsonl")
    write_jsonl(output_jsonl, submitted)
    print(f"submitted {len(submitted)} jobs; wrote {output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
