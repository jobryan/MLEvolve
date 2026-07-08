#!/usr/bin/env python3
"""Grade ablation submissions with the MLE-bench grader.

Worker mode grades the single run in --output-dir and writes
`<output-dir>/grader/grade_report.json` so the artifact sync captures it.
It never raises: failures produce a report with an `error` field and exit 0,
because grading problems must not turn a completed agent run into a failed job.

Backfill mode walks an artifacts root (one subdirectory per run) and grades
every submission it can find, writing a per-run grade_report.json plus a
consolidated grades JSONL keyed by run id for the results-export join.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VENDORED_MLEBENCH = ROOT / ".context/external/mle-bench"
GRADE_REPORT_SCHEMA_VERSION = "1.0.0"


def _import_mlebench():
    try:
        from mlebench.grade import grade_csv
        from mlebench.registry import registry
    except ImportError:
        if VENDORED_MLEBENCH.is_dir():
            sys.path.insert(0, str(VENDORED_MLEBENCH))
        from mlebench.grade import grade_csv
        from mlebench.registry import registry
    return grade_csv, registry


LFS_POINTER_PREFIX = b"version https://git-lfs"


def _repair_lfs_leaderboard(competition: Any) -> None:
    """Replace a git-lfs pointer leaderboard with the real CSV from the repo snapshot.

    The worker image pip-installs mle-bench outside the patch-overlay root, so a
    pointer-only leaderboard there is first fixed from the copy the worker patch
    tarball ships under the repo's vendored snapshot path. Some low-split tasks
    still have pointer-only vendored leaderboards; for those, use the worker's
    Kaggle credentials to refresh the leaderboard before grading.
    """
    leaderboard = getattr(competition, "leaderboard", None)
    if leaderboard is None:
        return
    leaderboard = Path(leaderboard)
    try:
        if not leaderboard.read_bytes().startswith(LFS_POINTER_PREFIX):
            return
    except OSError:
        return
    fallback = (
        VENDORED_MLEBENCH / "mlebench" / "competitions" / str(competition.id) / "leaderboard.csv"
    )
    try:
        if fallback.is_file() and not fallback.read_bytes().startswith(LFS_POINTER_PREFIX):
            leaderboard.write_bytes(fallback.read_bytes())
            return
    except OSError:
        return
    try:
        from mlebench.data import ensure_leaderboard_exists

        ensure_leaderboard_exists(competition, force=True)
    except Exception:
        return


AIS_FORK_ROOT = ROOT / ".context/external/AI-Scientist-v2-ablation"


def _read_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _recovery_metadata(output_dir: Path) -> dict[str, Any] | None:
    recovery_dir = output_dir / "ai_scientist_recovered_submission"
    recovery_report = _read_json_if_present(recovery_dir / "recovery_report.json")
    fallback_report = _read_json_if_present(recovery_dir / "fallback_report.json")
    if recovery_report is None and fallback_report is None:
        return None
    return {
        "recovery_report": recovery_report,
        "fallback_report": fallback_report,
        "fallback_used": bool(recovery_report and recovery_report.get("fallback_used")),
    }


def _best_persisted_ais_submission(output_dir: Path) -> Path | None:
    """Best contract-passing submission persisted by the fork's audit hook."""
    dest_dir = output_dir / "ais_submissions"
    index_path = dest_dir / "index.jsonl"
    if not dest_dir.is_dir():
        return None
    rows: list[dict[str, Any]] = []
    if index_path.is_file():
        for line in index_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    scored = [
        row
        for row in rows
        if row.get("metric_value") is not None and (dest_dir / f"{row['node_id']}.csv").is_file()
    ]
    if scored:
        lower_better = str(scored[0].get("metric_direction", "")).lower() == "minimize"
        best = min(scored, key=lambda r: r["metric_value"]) if lower_better else max(
            scored, key=lambda r: r["metric_value"]
        )
        return dest_dir / f"{best['node_id']}.csv"
    candidates = [p for p in dest_dir.glob("*.csv") if p.is_file()]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return None


def find_submission(output_dir: Path, system: str) -> Path | None:
    """Newest submission produced by the run, per system layout."""
    if system == "ai_scientist_v2":
        persisted = _best_persisted_ais_submission(output_dir)
        if persisted is not None:
            return persisted
    patterns_by_system = {
        "mlevolve": ["**/workspace/best_submission/submission.csv"],
        "ai_scientist_v2": [
            "ai_scientist_recovered_submission/submission.csv",
            "ai_scientist_experiments/**/submission.csv",
            "ai_scientist_workspaces/**/submission.csv",
        ],
    }
    patterns = patterns_by_system.get(system, []) + ["**/submission.csv"]
    for pattern in patterns:
        matches = [p for p in output_dir.glob(pattern) if p.is_file()]
        if matches:
            return max(matches, key=lambda p: p.stat().st_mtime)
    if system == "ai_scientist_v2":
        # The fork writes submissions under its live experiments/workspaces
        # trees; the worker's experiments-copy step does not always capture
        # them, so fall back to searching the fork checkout directly.
        fork_matches = [
            p
            for subdir in ("experiments", "workspaces")
            for p in (AIS_FORK_ROOT / subdir).glob("**/submission.csv")
            if p.is_file()
        ]
        if fork_matches:
            return max(fork_matches, key=lambda p: p.stat().st_mtime)
    return None


def grade_one(
    *,
    submission: Path | None,
    task_id: str,
    data_dir: Path,
    run_id: str | None,
    system: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": GRADE_REPORT_SCHEMA_VERSION,
        "grader": "mle-bench",
        "run_id": run_id,
        "system": system,
        "task_id": task_id,
        "submission_source": str(submission) if submission else None,
        "graded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "error": None,
        "report": None,
    }
    if submission is None:
        record["error"] = "no submission.csv found"
        return record
    try:
        grade_csv, registry = _import_mlebench()
        competition = registry.set_data_dir(data_dir).get_competition(task_id)
        _repair_lfs_leaderboard(competition)
        report = grade_csv(submission, competition)
        record["report"] = report.to_dict()
    except Exception as exc:  # noqa: BLE001 - grading must never kill the job
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def write_report(output_dir: Path, record: dict[str, Any], submission: Path | None) -> Path:
    grader_dir = output_dir / "grader"
    grader_dir.mkdir(parents=True, exist_ok=True)
    if submission is not None and submission.is_file():
        try:
            (grader_dir / "submission.csv").write_bytes(submission.read_bytes())
        except OSError as exc:
            record.setdefault("warnings", []).append(f"submission copy failed: {exc}")
    report_path = grader_dir / "grade_report.json"
    report_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report_path


def known_task_ids() -> list[str]:
    manifest_path = ROOT / "configs/ablations/tasks/mle_bench_lite.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = data.get("tasks", data if isinstance(data, list) else [])
    ids = [
        task.get("task_id") or task.get("id")
        for task in tasks
        if isinstance(task, dict) and (task.get("task_id") or task.get("id"))
    ]
    # Longest first so e.g. a hypothetical prefix task id cannot shadow a longer one.
    return sorted(set(ids), key=len, reverse=True)


def infer_run_metadata(run_dir_name: str, task_ids: list[str]) -> tuple[str | None, str]:
    slug = run_dir_name.replace("_", "-")
    task_id = next((tid for tid in task_ids if tid in slug), None)
    system = "ai_scientist_v2" if "ai-scientist-v2" in slug else "mlevolve"
    return task_id, system


def run_worker_mode(args: argparse.Namespace) -> int:
    output_dir = args.output_dir
    submission = find_submission(output_dir, args.system) if output_dir.is_dir() else None
    record = grade_one(
        submission=submission,
        task_id=args.task_id,
        data_dir=args.data_dir,
        run_id=args.run_id or os.environ.get("ABLATION_RUN_ID"),
        system=args.system,
    )
    if output_dir.is_dir() and args.system == "ai_scientist_v2":
        record["recovery"] = _recovery_metadata(output_dir)
    if not output_dir.is_dir():
        record["error"] = record["error"] or f"output dir missing: {output_dir}"
        print(json.dumps(record, sort_keys=True))
        return 0
    report_path = write_report(output_dir, record, submission)
    summary = record["report"] or {}
    print(
        f"grade_ablation_submissions: task={args.task_id} "
        f"score={summary.get('score')} valid={summary.get('valid_submission')} "
        f"error={record['error']} report={report_path}"
    )
    return 0


def run_backfill_mode(args: argparse.Namespace) -> int:
    task_ids = known_task_ids()
    rows: list[dict[str, Any]] = []
    run_dirs = sorted(p for p in args.artifacts_root.iterdir() if p.is_dir())
    for run_dir in run_dirs:
        task_id, system = infer_run_metadata(run_dir.name, task_ids)
        if task_id is None:
            rows.append(
                {
                    "run_id": run_dir.name,
                    "error": "could not infer task_id from directory name",
                }
            )
            continue
        submission = find_submission(run_dir, system)
        record = grade_one(
            submission=submission,
            task_id=task_id,
            data_dir=args.data_dir,
            run_id=run_dir.name,
            system=system,
        )
        if system == "ai_scientist_v2":
            record["recovery"] = _recovery_metadata(run_dir)
        write_report(run_dir, record, submission)
        report = record["report"] or {}
        recovery = record.get("recovery") or {}
        rows.append(
            {
                "run_id": run_dir.name,
                "task_id": task_id,
                "system": system,
                "score": report.get("score"),
                "valid_submission": report.get("valid_submission"),
                "any_medal": report.get("any_medal"),
                "above_median": report.get("above_median"),
                "is_lower_better": report.get("is_lower_better"),
                "submission_source": record["submission_source"],
                "fallback_used": recovery.get("fallback_used"),
                "error": record["error"],
            }
        )
    if args.grades_jsonl:
        args.grades_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.grades_jsonl.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    graded = sum(1 for row in rows if row.get("score") is not None)
    errors = sum(1 for row in rows if row.get("error"))
    print(
        f"grade_ablation_submissions backfill: runs={len(rows)} graded={graded} errors={errors}"
        + (f" grades_jsonl={args.grades_jsonl}" if args.grades_jsonl else "")
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_data_dir = os.environ.get("MLEBENCH_DATASET_DIR")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(default_data_dir) if default_data_dir else None,
        help="MLE-bench data dir containing <task>/prepared/{public,private}.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output-dir", type=Path, help="Worker mode: single run output dir.")
    mode.add_argument(
        "--artifacts-root", type=Path, help="Backfill mode: directory of per-run artifact dirs."
    )
    parser.add_argument("--task-id", help="Worker mode: MLE-bench competition id.")
    parser.add_argument("--system", default="mlevolve", help="Worker mode: mlevolve | ai_scientist_v2.")
    parser.add_argument("--run-id", help="Worker mode: run id (defaults to $ABLATION_RUN_ID).")
    parser.add_argument(
        "--grades-jsonl", type=Path, help="Backfill mode: consolidated grades JSONL output path."
    )
    args = parser.parse_args()
    if args.data_dir is None:
        parser.error("--data-dir or MLEBENCH_DATASET_DIR is required")
    if args.output_dir is not None and not args.task_id:
        parser.error("--task-id is required in worker mode")
    return args


def main() -> int:
    args = parse_args()
    if args.output_dir is not None:
        return run_worker_mode(args)
    return run_backfill_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
