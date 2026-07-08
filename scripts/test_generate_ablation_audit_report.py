#!/usr/bin/env python3
"""Tests for generate_ablation_audit_report.py."""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_REPORT = ROOT / "scripts/generate_ablation_audit_report.py"


def write_runs_csv(path: Path) -> None:
    rows = [
        {
            "system": "mlevolve",
            "variant_id": "eligible",
            "phase": "main",
            "seed": "1",
            "status": "success",
            "valid_submission": "True",
            "audit_status": "pass",
            "audit_failure_reasons": "",
            "normalized_score": "0.70",
        },
        {
            "system": "mlevolve",
            "variant_id": "eligible",
            "phase": "main",
            "seed": "2",
            "status": "success",
            "valid_submission": "True",
            "audit_status": "pass",
            "audit_failure_reasons": "",
            "normalized_score": "0.72",
        },
        {
            "system": "mlevolve",
            "variant_id": "high_score_audit_fail",
            "phase": "main",
            "seed": "1",
            "status": "success",
            "valid_submission": "True",
            "audit_status": "fail",
            "audit_failure_reasons": "forbidden_code_pattern",
            "normalized_score": "0.99",
        },
        {
            "system": "ai_scientist_v2",
            "variant_id": "invalid_only",
            "phase": "smoke",
            "seed": "1",
            "status": "invalid_submission",
            "valid_submission": "False",
            "audit_status": "fail",
            "audit_failure_reasons": "missing_submission",
            "normalized_score": "",
        },
        {
            "system": "ai_scientist_v2",
            "variant_id": "unaudited",
            "phase": "smoke",
            "seed": "1",
            "status": "success",
            "valid_submission": "True",
            "audit_status": "",
            "audit_failure_reasons": "",
            "normalized_score": "0.60",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_audit_promotion_gate() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        input_dir = tmp / "input"
        output_dir = tmp / "audit"
        input_dir.mkdir()
        write_runs_csv(input_dir / "runs.csv")

        result = subprocess.run(
            [
                sys.executable,
                str(AUDIT_REPORT),
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--min-confirmation-seeds",
                "2",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert result.returncode == 0, result.stderr

        rows = {row["variant_id"]: row for row in read_csv(output_dir / "promotion_gate.csv")}
        assert rows["eligible"]["promotion_decision"] == "promotion_eligible"
        assert rows["high_score_audit_fail"]["promotion_decision"] == "blocked_by_audit"
        assert rows["high_score_audit_fail"]["best_normalized_score"] == "0.99"
        assert rows["invalid_only"]["promotion_decision"] == "do_not_promote_invalid"
        assert rows["unaudited"]["promotion_decision"] == "provisional_unaudited"

        report = (output_dir / "audit_report.md").read_text(encoding="utf-8")
        assert "Audit-failed runs with non-invalid status: 1." in report
        assert "Invalid submission runs: 1." in report
        assert "`forbidden_code_pattern`: 1" in report
        assert "`missing_submission`: 1" in report


if __name__ == "__main__":
    test_audit_promotion_gate()
    print("generate_ablation_audit_report tests passed")
