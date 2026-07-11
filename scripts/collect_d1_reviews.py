#!/usr/bin/env python3
"""Collect Tranche-3 Block D1 (native research track) review outcomes.

Post-hoc walker over the D1 artifact prefix (S3 or a local mirror/fixture).
For each run it reads run_status.json (written by scripts/worker_native_d1.sh)
and the AI Scientist v2 native outputs inside
ai_scientist_experiments/<timestamp>_<idea>_attempt_<n>/:

  - review_text.txt        JSON review from perform_llm_review (Overall 1-10)
  - *.pdf                  generated paper (reflection/final PDFs)

and emits a summary CSV: idea, arm, attempt, run_id, completed, review_score,
paper_pdf_present.

Usage:
  python3 scripts/collect_d1_reviews.py \
    --root s3://autoresearch-experiments-058264252788-us-east-1/mlevolve-ai-scientist-v2-ablation/tranche3-d1 \
    --out t3d1_review_summary.csv

  python3 scripts/collect_d1_reviews.py --root /path/to/local/tranche3-d1 --out summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

CSV_FIELDS = [
    "idea",
    "arm",
    "attempt",
    "run_id",
    "completed",
    "review_score",
    "paper_pdf_present",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        required=True,
        help="D1 artifact root (s3://bucket/prefix/tranche3-d1 or a local dir); "
        "immediate children are per-run directories keyed by run_id.",
    )
    parser.add_argument("--out", required=True, help="Output CSV path.")
    return parser.parse_args()


def list_files_s3(root: str) -> dict[str, list[str]]:
    """Map run_id -> list of file keys relative to the run prefix."""
    import boto3

    bucket, prefix = root.removeprefix("s3://").split("/", 1)
    prefix = prefix.rstrip("/") + "/"
    s3 = boto3.client("s3")
    runs: dict[str, list[str]] = {}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"].removeprefix(prefix)
            if "/" not in rel:
                continue
            run_id, run_rel = rel.split("/", 1)
            runs.setdefault(run_id, []).append(run_rel)
    return runs


def read_file_s3(root: str, run_id: str, run_rel: str) -> str:
    import boto3

    bucket, prefix = root.removeprefix("s3://").split("/", 1)
    key = f"{prefix.rstrip('/')}/{run_id}/{run_rel}"
    body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"]
    return body.read().decode("utf-8", errors="replace")


def list_files_local(root: str) -> dict[str, list[str]]:
    base = Path(root)
    runs: dict[str, list[str]] = {}
    if not base.is_dir():
        return runs
    for run_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        runs[run_dir.name] = [
            f.relative_to(run_dir).as_posix() for f in sorted(run_dir.rglob("*")) if f.is_file()
        ]
    return runs


def read_file_local(root: str, run_id: str, run_rel: str) -> str:
    return (Path(root) / run_id / run_rel).read_text(encoding="utf-8", errors="replace")


def extract_review_score(review_json_text: str) -> float | None:
    try:
        review = json.loads(review_json_text)
    except json.JSONDecodeError:
        return None
    score = review.get("Overall") if isinstance(review, dict) else None
    if isinstance(score, (int, float)):
        return float(score)
    return None


def summarize_run(run_id: str, files: list[str], read_file) -> dict:
    row: dict = {
        "idea": "",
        "arm": "",
        "attempt": "",
        "run_id": run_id,
        "completed": False,
        "review_score": "",
        "paper_pdf_present": False,
    }

    if "run_status.json" in files:
        try:
            status = json.loads(read_file(run_id, "run_status.json"))
            row["idea"] = status.get("idea_slug") or status.get("idea_path", "")
            row["arm"] = status.get("arm", "")
            row["attempt"] = status.get("attempt_id", "")
            row["completed"] = bool(status.get("completed"))
        except (json.JSONDecodeError, OSError):
            pass

    experiment_files = [
        f for f in files if f.startswith("ai_scientist_experiments/")
    ]
    row["paper_pdf_present"] = any(f.endswith(".pdf") for f in experiment_files)

    review_files = sorted(
        f for f in experiment_files if f.endswith("/review_text.txt")
    )
    for review_file in review_files:
        score = extract_review_score(read_file(run_id, review_file))
        if score is not None:
            row["review_score"] = score
            break

    return row


def main() -> int:
    args = parse_args()
    if args.root.startswith("s3://"):
        runs = list_files_s3(args.root)
        read_file = lambda run_id, rel: read_file_s3(args.root, run_id, rel)  # noqa: E731
    else:
        runs = list_files_local(args.root)
        read_file = lambda run_id, rel: read_file_local(args.root, run_id, rel)  # noqa: E731

    rows = [
        summarize_run(run_id, files, read_file)
        for run_id, files in sorted(runs.items())
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    completed = sum(1 for r in rows if r["completed"])
    with_paper = sum(1 for r in rows if r["paper_pdf_present"])
    with_review = sum(1 for r in rows if r["review_score"] != "")
    print(
        f"wrote {len(rows)} rows -> {out_path} "
        f"(completed={completed}, paper_pdf={with_paper}, review_score={with_review})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
