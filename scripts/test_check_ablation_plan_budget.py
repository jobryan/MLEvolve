#!/usr/bin/env python3
"""Tests for check_ablation_plan_budget.py."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_ablation_plan_budget.py"
spec = importlib.util.spec_from_file_location("check_ablation_plan_budget", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def manifest(run_id: str, phase: str, system: str, cost: float) -> dict:
    return {
        "run_id": run_id,
        "phase": phase,
        "system": system,
        "budget": {"max_cost_usd": cost},
    }


def test_summarize_plan_groups_costs() -> None:
    summary = module.summarize_plan(
        [
            manifest("run-1", "smoke", "mlevolve", 50),
            manifest("run-2", "smoke", "ai_scientist_v2", 50),
            manifest("run-3", "screening", "mlevolve", 150),
        ],
        spend_cap_usd=300,
    )
    assert summary["run_count"] == 3
    assert summary["total_max_cost_usd"] == 250
    assert summary["remaining_cap_usd"] == 50
    assert summary["over_cap"] is False
    assert summary["by_phase"]["smoke"] == {"count": 2, "max_cost_usd": 100}
    assert summary["by_system"]["mlevolve"] == {"count": 2, "max_cost_usd": 200}


def test_summarize_plan_flags_over_cap() -> None:
    summary = module.summarize_plan([manifest("run-1", "screening", "mlevolve", 150)], spend_cap_usd=100)
    assert summary["over_cap"] is True
    assert summary["remaining_cap_usd"] == -50


def test_cli_exits_nonzero_when_over_cap() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        path = tmp / "manifests.jsonl"
        path.write_text(json.dumps(manifest("run-1", "smoke", "mlevolve", 50)) + "\n", encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--manifest-jsonl",
                str(path),
                "--spend-cap-usd",
                "25",
            ],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert result.returncode == 2
        assert json.loads(result.stdout)["over_cap"] is True


if __name__ == "__main__":
    test_summarize_plan_groups_costs()
    test_summarize_plan_flags_over_cap()
    test_cli_exits_nonzero_when_over_cap()
    print("check_ablation_plan_budget tests passed")
