#!/usr/bin/env python3
"""Run first-pass audit checks on an ablation submission."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


FORBIDDEN_CODE_PATTERNS = [
    re.compile(r"prepared/private", re.IGNORECASE),
    re.compile(r"answers\.csv", re.IGNORECASE),
    re.compile(r"gold_submission", re.IGNORECASE),
    re.compile(r"leaderboard\.csv", re.IGNORECASE),
    re.compile(r"private/test\.csv", re.IGNORECASE),
    re.compile(r"/home/data/private", re.IGNORECASE),
    re.compile(r"read_csv\s*\(\s*['\"]/?home/submission/submission\.csv", re.IGNORECASE),
]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def looks_numeric(values: list[str]) -> bool:
    for value in values:
        try:
            float(value)
        except (TypeError, ValueError):
            return False
    return bool(values)


def has_non_finite(values: list[str]) -> bool:
    for value in values:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return True
        if not math.isfinite(parsed):
            return True
    return False


def audit_submission(
    run_id: str,
    node_id: str,
    submission: Path,
    sample_submission: Path,
    code: Path | None,
    max_runtime_seconds: float | None,
    actual_runtime_seconds: float | None,
) -> dict[str, Any]:
    blocking: list[str] = []
    report_only: list[str] = []
    checks: dict[str, Any] = {}

    if not submission.exists():
        blocking.append("missing_submission")
    if not sample_submission.exists():
        blocking.append("missing_sample_submission")

    if not blocking:
        sample_columns, sample_rows = read_csv(sample_submission)
        submission_columns, submission_rows = read_csv(submission)
        checks["sample_columns"] = sample_columns
        checks["submission_columns"] = submission_columns
        checks["sample_row_count"] = len(sample_rows)
        checks["submission_row_count"] = len(submission_rows)

        if submission_columns != sample_columns:
            blocking.append("column_mismatch")
        if len(submission_rows) != len(sample_rows):
            blocking.append("row_count_mismatch")
        if not submission_rows:
            blocking.append("empty_submission")

        for column in sample_columns:
            sample_values = [row.get(column, "") for row in sample_rows]
            submission_values = [row.get(column, "") for row in submission_rows]
            if looks_numeric(sample_values) and has_non_finite(submission_values):
                blocking.append("non_finite_numeric_value")
                break

        if submission_rows and len({tuple(row.items()) for row in submission_rows}) == 1:
            report_only.append("all_submission_rows_identical")

    if (
        max_runtime_seconds is not None
        and actual_runtime_seconds is not None
        and actual_runtime_seconds > max_runtime_seconds
    ):
        blocking.append("runtime_over_limit")
        checks["max_runtime_seconds"] = max_runtime_seconds
        checks["actual_runtime_seconds"] = actual_runtime_seconds

    if code is not None and code.exists():
        code_text = code.read_text(encoding="utf-8", errors="replace")
        matches = [pattern.pattern for pattern in FORBIDDEN_CODE_PATTERNS if pattern.search(code_text)]
        if matches:
            blocking.append("forbidden_code_pattern")
            checks["forbidden_code_patterns"] = matches
        if re.search(r"https?://", code_text):
            report_only.append("external_url_reference")
    elif code is not None:
        report_only.append("missing_code_for_audit")

    return {
        "run_id": run_id,
        "node_id": node_id,
        "status": "fail" if blocking else "pass",
        "blocking_failures": sorted(set(blocking)),
        "report_only_findings": sorted(set(report_only)),
        "checks": checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--submission", required=True, type=Path)
    parser.add_argument("--sample-submission", required=True, type=Path)
    parser.add_argument("--code", type=Path)
    parser.add_argument("--max-runtime-seconds", type=float)
    parser.add_argument("--actual-runtime-seconds", type=float)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit_submission(
        run_id=args.run_id,
        node_id=args.node_id,
        submission=args.submission,
        sample_submission=args.sample_submission,
        code=args.code,
        max_runtime_seconds=args.max_runtime_seconds,
        actual_runtime_seconds=args.actual_runtime_seconds,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
