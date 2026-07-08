#!/usr/bin/env python3
"""Recover an AI Scientist MLE-bench submission from saved best-solution code.

AI Scientist's tree-search artifacts preserve the generated best-solution code
and validation metrics, but the execution workspace containing
``working/submission.csv`` is not always copied into the experiment artifact.
On Batch workers we can safely re-run the selected best solution against the
same prepared public data before grading, then copy the recovered submission
into the run output directory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_value(node: dict[str, Any]) -> float | None:
    metric = node.get("metric")
    if isinstance(metric, dict):
        metric = metric.get("value")
    if not isinstance(metric, dict):
        return None
    metric_names = metric.get("metric_names")
    if not isinstance(metric_names, list):
        return None
    for metric_record in metric_names:
        if not isinstance(metric_record, dict):
            continue
        data = metric_record.get("data")
        if not isinstance(data, list) or not data:
            continue
        first = data[0]
        if not isinstance(first, dict):
            continue
        value = first.get("final_value")
        if isinstance(value, int | float) and value == value:
            return float(value)
    return None


def lower_is_better(node: dict[str, Any], default: bool = True) -> bool:
    metric = node.get("metric")
    if isinstance(metric, dict):
        metric = metric.get("value")
    if isinstance(metric, dict):
        metric_names = metric.get("metric_names")
        if isinstance(metric_names, list) and metric_names:
            first = metric_names[0]
            if isinstance(first, dict) and isinstance(first.get("lower_is_better"), bool):
                return bool(first["lower_is_better"])
    return default


def best_node_id_from_journal(journal_path: Path) -> str | None:
    try:
        journal = read_json(journal_path)
    except Exception:
        return None
    nodes = journal.get("nodes") if isinstance(journal, dict) else journal
    if not isinstance(nodes, list):
        return None
    candidates: list[tuple[float, str]] = []
    for node in nodes:
        if not isinstance(node, dict) or node.get("is_buggy") is True:
            continue
        value = metric_value(node)
        node_id = node.get("id")
        if value is None or not node_id:
            continue
        score = value if lower_is_better(node) else -value
        candidates.append((score, str(node_id)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def candidate_solutions(output_dir: Path) -> list[Path]:
    paths = [path for path in output_dir.glob("ai_scientist_experiments/**/best_solution_*.py") if path.is_file()]
    if not paths:
        return []

    ranked: list[tuple[int, float, Path]] = []
    for path in paths:
        node_id = path.stem.removeprefix("best_solution_")
        journal = path.with_name("journal.json")
        rank = 1
        best_id = best_node_id_from_journal(journal) if journal.is_file() else None
        if best_id and best_id == node_id:
            rank = 0
        ranked.append((rank, -path.stat().st_mtime, path))
    ranked.sort()
    return [item[2] for item in ranked]


def find_submission(work_dir: Path) -> Path | None:
    preferred = work_dir / "working" / "submission.csv"
    if preferred.is_file():
        return preferred
    matches = [path for path in work_dir.glob("**/submission.csv") if path.is_file()]
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def sample_submission_path(data_dir: Path, task_id: str) -> Path | None:
    public_dir = data_dir / task_id / "prepared" / "public"
    for filename in ("sample_submission.csv", "sampleSubmission.csv", "sample_submission_null.csv"):
        candidate = public_dir / filename
        if candidate.is_file():
            return candidate
    return None


def write_sample_fallback(args: argparse.Namespace, recovery_dir: Path, record: dict[str, Any]) -> Path | None:
    sample = sample_submission_path(args.data_dir, args.task_id)
    if sample is None:
        record["fallback_error"] = "sample submission not found"
        return None

    copied = recovery_dir / "submission.csv"
    shutil.copy2(sample, copied)
    fallback_report = {
        "schema_version": "1.0.0",
        "run_id": args.run_id,
        "task_id": args.task_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "sample_submission_copy",
        "source": str(sample),
        "submission_path": str(copied),
        "notes": (
            "Emergency harness fallback used only after AI Scientist produced no "
            "recoverable submission. Treat score as a validity floor, not as an "
            "agent-owned solution."
        ),
    }
    (recovery_dir / "fallback_report.json").write_text(
        json.dumps(fallback_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    record["fallback_used"] = True
    record["fallback_method"] = "sample_submission_copy"
    record["fallback_source"] = str(sample)
    return copied


def recover(args: argparse.Namespace) -> int:
    output_dir = args.output_dir
    recovery_dir = output_dir / "ai_scientist_recovered_submission"
    recovery_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "schema_version": "1.0.0",
        "run_id": args.run_id,
        "task_id": args.task_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "attempted": False,
        "solution_path": None,
        "submission_path": None,
        "returncode": None,
        "error": None,
        "fallback_used": False,
        "fallback_method": None,
        "fallback_source": None,
        "fallback_error": None,
    }

    candidates = candidate_solutions(output_dir)
    if not candidates:
        record["error"] = "no best_solution_*.py found"
        fallback = write_sample_fallback(args, recovery_dir, record) if args.allow_fallback else None
        if fallback is not None:
            record["submission_path"] = str(fallback)
        (recovery_dir / "recovery_report.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(record["error"])
        return 0

    solution = candidates[0]
    record["attempted"] = True
    record["solution_path"] = str(solution)
    work_dir = recovery_dir / "work"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    env = os.environ.copy()
    env["MLEBENCH_DATASET_DIR"] = str(args.data_dir)
    env["ABLATION_TASK_ID"] = args.task_id
    env.setdefault(
        "ABLATION_SAMPLE_SUBMISSION_PATH",
        str(args.data_dir / args.task_id / "prepared" / "public" / "sample_submission.csv"),
    )

    stdout_path = recovery_dir / "stdout.txt"
    stderr_path = recovery_dir / "stderr.txt"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        try:
            completed = subprocess.run(
                [sys.executable, str(solution.resolve())],
                cwd=work_dir,
                env=env,
                stdout=stdout,
                stderr=stderr,
                timeout=args.timeout_seconds,
                check=False,
            )
            record["returncode"] = completed.returncode
        except subprocess.TimeoutExpired:
            record["error"] = f"timeout after {args.timeout_seconds}s"

    submission = find_submission(work_dir)
    if submission is None and record["error"] is None:
        record["error"] = "no submission.csv found after executing best solution"
    if submission is not None:
        copied = recovery_dir / "submission.csv"
        shutil.copy2(submission, copied)
        record["submission_path"] = str(copied)
    elif args.allow_fallback:
        fallback = write_sample_fallback(args, recovery_dir, record)
        if fallback is not None:
            record["submission_path"] = str(fallback)

    (recovery_dir / "recovery_report.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Copy the task sample submission as a labeled validity-floor fallback if normal recovery fails.",
    )
    return parser.parse_args()


def main() -> int:
    return recover(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
