"""Re-grade already-synced runs on a worker (leaderboard-fix backfill).

Runs inside the worker patch root. For each run id: downloads the run's synced
best_submission/submission.csv from S3, grades it against held-out answers with
the harness grading script (which now sees real leaderboard CSVs), and uploads
the refreshed grade_report.json back over the run's grader/ prefix.

Usage (one job per task; dataset is prepared once):
    python3 experience/regrade_runs.py --task-id leaf-classification \
        --run-ids rid1,rid2,rid3 \
        --artifact-root s3://<bucket>/<prefix>/artifacts --phase screening
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--run-ids", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--data-dir", default=os.environ.get("MLEBENCH_DATASET_DIR", ""))
    args = parser.parse_args(argv)
    if not args.data_dir:
        raise SystemExit("MLEBENCH_DATASET_DIR required")

    import boto3

    s3 = boto3.client("s3")
    bucket, root_key = args.artifact_root.removeprefix("s3://").split("/", 1)

    desc = Path(args.data_dir) / args.task_id / "prepared/public/description.md"
    if not desc.exists():
        r = sh(["mlebench", "prepare", "-c", args.task_id, "--data-dir", args.data_dir])
        if r.returncode != 0:
            raise SystemExit(f"prepare failed: {r.stderr[-400:]}")

    results = {}
    for rid in args.run_ids.split(","):
        rid = rid.strip()
        prefix = f"{root_key}/{args.phase}/{rid}/"
        sub_key = None
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith("best_submission/submission.csv"):
                    sub_key = obj["Key"]
                    break
            if sub_key:
                break
        if not sub_key:
            results[rid] = "no-submission-synced"
            continue

        outdir = ROOT / ".context/e2/regrade" / rid
        subdir = outdir / "workspace/best_submission"
        subdir.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, sub_key, str(subdir / "submission.csv"))

        r = sh([sys.executable, "scripts/grade_ablation_submissions.py",
                "--output-dir", str(outdir), "--task-id", args.task_id,
                "--system", "mlevolve", "--run-id", rid, "--data-dir", args.data_dir],
               cwd=ROOT)
        report_path = outdir / "grader/grade_report.json"
        if report_path.exists():
            s3.upload_file(str(report_path), bucket, f"{prefix}grader/grade_report.json")
            rep = json.loads(report_path.read_text())
            score = rep["report"].get("score") if rep.get("report") else None
            results[rid] = f"regraded score={score} err={rep.get('error')}"
        else:
            results[rid] = f"grade-script-failed rc={r.returncode} {r.stderr[-150:]}"

    for rid, res in results.items():
        print(rid[:70], "->", res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
