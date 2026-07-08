#!/usr/bin/env python3
"""Tests for export_ablation_results.py."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPORTER = ROOT / "scripts/export_ablation_results.py"
TASK_MANIFEST = ROOT / "configs/ablations/tasks/mle_bench_lite.json"


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def base_run(run_id: str, system: str, variant: str, status: str, final_score: float | None) -> dict:
    valid = status == "success"
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "system": system,
        "benchmark_track": "shared_mle_bench_lite",
        "task_id": "spooky-author-identification",
        "task_manifest_version": "mle-bench-lite-v1",
        "variant_id": variant,
        "phase": "smoke",
        "seed": 1,
        "status": status,
        "started_at": "2026-06-23T10:00:00Z",
        "completed_at": "2026-06-23T10:30:00Z",
        "budget": {"wall_time_seconds": 14400, "max_nodes": 10, "max_cost_usd": 50.0},
        "actual": {"wall_time_seconds": 1800, "nodes": 2 if valid else 1, "cost_usd": 3.25},
        "grader": {"name": "mle-bench", "version": "test", "command": "mlebench grade"},
        "audit_summary": {
            "status": "pass" if valid else "fail",
            "failure_reasons": [] if valid else ["missing_submission"],
        },
        "metrics": {
            "best_validation_score": final_score,
            "final_score": final_score,
            "normalized_score": 0.8 if valid else None,
            "valid_submission": valid,
        },
        "artifacts": {"runs_path": f"runs/{run_id}/run.json", "nodes_path": f"runs/{run_id}/nodes.jsonl"},
        "error": None
        if valid
        else {"type": "invalid_submission", "message": "submission.csv was not produced"},
    }


def base_node(
    node_id: str,
    run_id: str,
    system: str,
    variant: str,
    operator: str,
    status: str,
    parents: list[str] | None = None,
) -> dict:
    success = status == "success"
    return {
        "schema_version": "1.0",
        "node_id": node_id,
        "run_id": run_id,
        "system": system,
        "task_id": "spooky-author-identification",
        "variant_id": variant,
        "seed": 1,
        "operator": operator,
        "stage": operator.lower(),
        "parent_node_ids": parents or [],
        "reference_node_ids": [],
        "branch_id": "branch-1",
        "depth": len(parents or []),
        "status": status,
        "created_at": "2026-06-23T10:01:00Z",
        "completed_at": "2026-06-23T10:08:00Z",
        "prompt_paths": [],
        "generated_code_path": f"runs/{run_id}/code/{node_id}.py",
        "submission_path": f"runs/{run_id}/submissions/{node_id}/submission.csv" if success else None,
        "grader_output_path": None,
        "audit_output_path": None,
        "metric_name": "multi-class-log-loss",
        "metric_direction": "minimize",
        "validation_score": 0.41 if success else None,
        "final_score": 0.42 if success else None,
        "normalized_score": 0.8 if success else None,
        "wall_time_seconds": 420,
        "cost": {"usd": 1.1},
        "token_usage": {"input_tokens": 1000, "output_tokens": 300},
        "memory_sources": {"child_history": [], "global_retrieval": [], "journal_summary": None, "coldstart": []},
        "selection": {"policy": "test_policy", "selection_score": 0.8},
        "diversity": {"strategy_labels": ["linear"], "nearest_neighbor_distance": 0.25},
        "artifacts": {},
        "error": None
        if success
        else {"type": "missing_submission", "message": "node did not write submission.csv"},
    }


def run_export(
    tmp: Path,
    runs: list[dict],
    nodes: list[dict],
    grades: list[dict] | None = None,
) -> subprocess.CompletedProcess[str]:
    runs_path = tmp / "runs.jsonl"
    nodes_path = tmp / "nodes.jsonl"
    output_dir = tmp / "export"
    write_jsonl(runs_path, runs)
    write_jsonl(nodes_path, nodes)
    command = [
        sys.executable,
        str(EXPORTER),
        "--runs-jsonl",
        str(runs_path),
        "--nodes-jsonl",
        str(nodes_path),
        "--task-manifest",
        str(TASK_MANIFEST),
        "--output-dir",
        str(output_dir),
    ]
    if grades is not None:
        grades_path = tmp / "grades.jsonl"
        write_jsonl(grades_path, grades)
        command += ["--grades-jsonl", str(grades_path)]
    return subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_successful_export() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        runs = [
            base_run("run-mlevolve", "mlevolve", "default_mcgs", "success", 0.42),
            base_run("run-ais", "ai_scientist_v2", "stage_as_operator_default", "invalid_submission", None),
        ]
        nodes = [
            base_node("node-1", "run-mlevolve", "mlevolve", "default_mcgs", "Draft", "success"),
            base_node("node-2", "run-mlevolve", "mlevolve", "default_mcgs", "Debug", "success", ["node-1"]),
            base_node("node-3", "run-ais", "ai_scientist_v2", "stage_as_operator_default", "Draft", "invalid_submission"),
        ]
        result = run_export(tmp, runs, nodes)
        assert result.returncode == 0, result.stderr

        output_dir = tmp / "export"
        for filename in [
            "runs.csv",
            "nodes.csv",
            "operator_stats.csv",
            "variant_summary.csv",
            "audit_summary.csv",
            "data_quality.json",
        ]:
            assert (output_dir / filename).exists(), filename

        quality = json.loads((output_dir / "data_quality.json").read_text(encoding="utf-8"))
        assert quality["status"] == "pass"
        assert quality["run_count"] == 2
        assert quality["node_count"] == 3

        variant_rows = read_csv(output_dir / "variant_summary.csv")
        by_variant = {row["variant_id"]: row for row in variant_rows}
        assert by_variant["default_mcgs"]["valid_submission_rate"] == "1.0"
        assert by_variant["stage_as_operator_default"]["valid_submission_rate"] == "0.0"

        operator_rows = read_csv(output_dir / "operator_stats.csv")
        assert {row["operator"] for row in operator_rows} == {"Debug", "Draft"}

        audit_rows = read_csv(output_dir / "audit_summary.csv")
        assert any(row["failure_reason"] == "missing_submission" for row in audit_rows)


def test_grades_join_overrides_final_score() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        runs = [
            base_run("run-mlevolve", "mlevolve", "default_mcgs", "success", None),
        ]
        nodes = [
            base_node("node-1", "run-mlevolve", "mlevolve", "default_mcgs", "Draft", "success"),
        ]
        grades = [
            {
                "run_id": "run_mlevolve",  # underscore form must still join
                "task_id": "spooky-author-identification",
                "score": 0.3111,
                "valid_submission": True,
                "any_medal": False,
                "above_median": True,
                "is_lower_better": True,
            },
            {"run_id": "run-unknown", "score": 0.9},
        ]
        result = run_export(tmp, runs, nodes, grades=grades)
        assert result.returncode == 0, result.stderr
        assert "grades: rows=2 matched_runs=1" in result.stdout

        run_rows = read_csv(tmp / "export/runs.csv")
        assert len(run_rows) == 1
        assert run_rows[0]["final_score"] == "0.3111"


def test_export_rejects_bad_ids() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        runs = [
            base_run("run-dup", "mlevolve", "default_mcgs", "success", 0.42),
            base_run("run-dup", "mlevolve", "default_mcgs", "success", 0.43),
        ]
        nodes = [
            base_node("node-1", "run-dup", "mlevolve", "default_mcgs", "Draft", "success", ["missing-parent"])
        ]
        result = run_export(tmp, runs, nodes)
        assert result.returncode == 1

        quality = json.loads((tmp / "export/data_quality.json").read_text(encoding="utf-8"))
        assert quality["status"] == "fail"
        assert any("duplicate run_id" in error for error in quality["errors"])
        assert any("missing parent node" in error for error in quality["errors"])


if __name__ == "__main__":
    test_successful_export()
    test_grades_join_overrides_final_score()
    test_export_rejects_bad_ids()
    print("export_ablation_results tests passed")
