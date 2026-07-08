#!/usr/bin/env python3
"""Tests for archive_mlevolve_logs.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVER = ROOT / "scripts/archive_mlevolve_logs.py"


def test_archive_mlevolve_logs_dry_run() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        log_dir = tmp / "runs/example/logs"
        log_dir.mkdir(parents=True)
        keep_dir = tmp / "runs/small/logs"
        keep_dir.mkdir(parents=True)

        (log_dir / "MLEvolve.log").write_text("x" * 8, encoding="utf-8")
        (log_dir / "MLEvolve.verbose.log").write_text("y" * 8, encoding="utf-8")
        (keep_dir / "MLEvolve.log").write_text("z", encoding="utf-8")

        result = subprocess.run(
            [
                sys.executable,
                str(ARCHIVER),
                "--root",
                str(tmp),
                "--destination-uri",
                "s3://bucket/prefix",
                "--threshold-bytes",
                "10",
                "--dry-run",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        payload = json.loads(result.stdout)
        assert len(payload["groups"]) == 1
        uploads = payload["groups"][0]["uploads"]
        assert {Path(upload["path"]).name for upload in uploads} == {
            "MLEvolve.log",
            "MLEvolve.verbose.log",
        }
        assert all(upload["s3_uri"].startswith("s3://bucket/prefix/") for upload in uploads)
        assert (log_dir / "MLEvolve.log").exists()
        assert (log_dir / "MLEvolve.verbose.log").exists()


if __name__ == "__main__":
    test_archive_mlevolve_logs_dry_run()
    print("archive_mlevolve_logs tests passed")
