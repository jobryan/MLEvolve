#!/usr/bin/env python3
"""Tests for analyze_ablation_results.py."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPORTER = ROOT / "scripts/export_ablation_results.py"
ANALYZER = ROOT / "scripts/analyze_ablation_results.py"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_sample_export(export_dir: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(EXPORTER),
            "--runs-jsonl",
            str(ROOT / "configs/ablations/schema/sample_runs.jsonl"),
            "--nodes-jsonl",
            str(ROOT / "configs/ablations/schema/sample_nodes.jsonl"),
            "--output-dir",
            str(export_dir),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0, result.stderr


def test_analysis_outputs() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        export_dir = tmp / "export"
        analysis_dir = tmp / "analysis"
        run_sample_export(export_dir)

        result = subprocess.run(
            [
                sys.executable,
                str(ANALYZER),
                "--input-dir",
                str(export_dir),
                "--output-dir",
                str(analysis_dir),
                "--bootstrap-iterations",
                "100",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert result.returncode == 0, result.stderr

        expected_files = [
            "variant_metrics.csv",
            "pairwise_win_rates.csv",
            "bootstrap_intervals.csv",
            "validation_gap.csv",
            "diversity_summary.csv",
            "draft_recommendations.csv",
            "analysis_summary.json",
            "ablation_analysis.md",
            "plots/variant_effective_score.svg",
            "plots/cost_quality_frontier.svg",
            "plots/operator_success.svg",
            "plots/diversity_vs_performance.svg",
        ]
        for filename in expected_files:
            assert (analysis_dir / filename).exists(), filename

        metrics = {row["variant_id"]: row for row in read_csv(analysis_dir / "variant_metrics.csv")}
        assert metrics["default_mcgs"]["valid_submission_rate"] == "1.0"
        assert metrics["stage_as_operator_default"]["valid_submission_rate"] == "0.0"
        assert metrics["default_mcgs"]["mean_effective_normalized_score"] == "0.74"
        assert metrics["stage_as_operator_default"]["mean_effective_normalized_score"] == "0.0"
        assert metrics["default_mcgs"]["node_count"] == "1"
        assert metrics["default_mcgs"]["valid_node_count"] == "1"
        assert metrics["stage_as_operator_default"]["node_count"] == "1"
        assert metrics["stage_as_operator_default"]["valid_node_count"] == "0"

        pairwise = read_csv(analysis_dir / "pairwise_win_rates.csv")
        assert len(pairwise) == 1
        assert pairwise[0]["paired_count"] == "1"
        assert pairwise[0]["a_win_rate"] == "0.0"
        assert pairwise[0]["b_wins"] == "1"

        recs = {row["variant_id"]: row for row in read_csv(analysis_dir / "draft_recommendations.csv")}
        assert recs["default_mcgs"]["draft_decision"] == "needs_more_evidence"
        assert recs["stage_as_operator_default"]["draft_decision"] == "repair_or_drop"

        summary = json.loads((analysis_dir / "analysis_summary.json").read_text(encoding="utf-8"))
        assert summary["run_count"] == 2
        assert summary["node_count"] == 2
        assert summary["variant_count"] == 2


if __name__ == "__main__":
    test_analysis_outputs()
    print("analyze_ablation_results tests passed")
