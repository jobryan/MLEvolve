#!/usr/bin/env python3
"""Tests for build_ablation_subset.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_ablation_subset.py"


def manifest(system: str, variant_id: str) -> dict:
    run_id = f"screening-{system}-{variant_id}-tabular-seed-1"
    return {
        "run_id": run_id,
        "system": system,
        "variant_id": variant_id,
        "task_id": "tabular-playground-series-may-2022",
        "phase": "screening",
        "seed": 1,
        "status": "planned",
        "budget": {
            "max_cost_usd": 150.0,
            "max_nodes": 30,
            "wall_time_seconds": 14400,
        },
        "config_overrides": {},
        "artifacts": {
            "manifest_path": f".context/ablation/run_manifests/{run_id}.json",
            "output_dir": f".context/ablation/runs/{run_id}",
        },
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")


def test_subset_clone_budget_and_controls() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        source = tmp / "source.jsonl"
        output = tmp / "subset.jsonl"
        output_root = tmp / "manifests"
        write_jsonl(
            source,
            [
                manifest("mlevolve", "default_mcgs"),
                manifest("ai_scientist_v2", "stage_as_operator_default"),
            ],
        )

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--source-jsonl",
                str(source),
                "--output-jsonl",
                str(output),
                "--output-root",
                str(output_root),
                "--run-id-prefix",
                "micro-smoke",
                "--task",
                "tabular-playground-series-may-2022",
                "--max-cost-usd",
                "40",
                "--wall-time-seconds",
                "1800",
                "--max-nodes",
                "2",
                "--force",
            ],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert result.returncode == 0, result.stderr
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 2
        assert all(row["run_id"].startswith("micro-smoke-") for row in rows)
        assert all(row["phase"] == "smoke" for row in rows)
        assert all(row["budget"]["max_cost_usd"] == 40.0 for row in rows)
        assert all(row["budget"]["wall_time_seconds"] == 1800 for row in rows)
        mlevolve = next(row for row in rows if row["system"] == "mlevolve")
        assert mlevolve["config_overrides"]["agent.initial_drafts"] == 1
        ai_scientist = next(row for row in rows if row["system"] == "ai_scientist_v2")
        assert ai_scientist["runtime_controls"]["ai_scientist_budget_patch"] is True


if __name__ == "__main__":
    test_subset_clone_budget_and_controls()
    print("build_ablation_subset tests passed")
