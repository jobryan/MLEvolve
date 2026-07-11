#!/usr/bin/env python3
"""Tests for Tranche-3 Block D1 native-track job specs and tooling.

Standalone (not appended to test_aws_worker_jobs.py: D1 is outside the
MLE-bench manifest schema those tests are built around).
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOBS_PATH = ROOT / ".context/ablation/aws_jobs/t3d1_jobs_spot24.jsonl"
SUMMARY_PATH = ROOT / ".context/ablation/aws_jobs/t3d1_build_summary.json"
WORKER_SH = ROOT / "scripts/worker_native_d1.sh"
COLLECTOR = ROOT / "scripts/collect_d1_reviews.py"

FULL_STAGES = {"T3_AIS_STAGE1": "6", "T3_AIS_STAGE2": "2", "T3_AIS_STAGE3": "6", "T3_AIS_STAGE4": "4"}
DDIO_STAGES = {"T3_AIS_STAGE1": "12", "T3_AIS_STAGE2": "6", "T3_AIS_STAGE3": "0", "T3_AIS_STAGE4": "0"}


def load_jobs() -> list[dict]:
    return [json.loads(line) for line in JOBS_PATH.read_text().splitlines() if line.strip()]


def env_map(job: dict) -> dict[str, str]:
    return {e["name"]: e["value"] for e in job["containerOverrides"]["environment"]}


def test_job_count_and_matrix() -> None:
    jobs = load_jobs()
    assert len(jobs) == 18, f"expected 18 jobs, got {len(jobs)}"
    names = [j["jobName"] for j in jobs]
    assert len(set(names)) == 18, "duplicate jobName"
    arms = {env_map(j)["ABLATION_VARIANT_ID"] for j in jobs}
    assert arms == {"full_pipeline", "draft_debug_improve_only"}
    # 2 arms x 3 ideas x attempts 1-3
    combos = {
        (env_map(j)["ABLATION_VARIANT_ID"], env_map(j)["T3D1_IDEA_IDX"], env_map(j)["T3D1_ATTEMPT_ID"])
        for j in jobs
    }
    assert len(combos) == 18
    assert {c[1] for c in combos} == {"0", "1", "2"}
    assert {c[2] for c in combos} == {"1", "2", "3"}


def test_env_correctness_per_arm() -> None:
    for job in load_jobs():
        env = env_map(job)
        arm = env["ABLATION_VARIANT_ID"]
        stages = {k: env[k] for k in FULL_STAGES}
        if arm == "full_pipeline":
            assert stages == FULL_STAGES, f"{job['jobName']}: {stages}"
            assert "AIS_DISABLED_STAGES" not in env
        else:
            assert stages == DDIO_STAGES, f"{job['jobName']}: {stages}"
            assert env["AIS_DISABLED_STAGES"] == "3,4"
        # Budget parity + runtime clamp shared by both arms.
        assert env["T3_AIS_STEPS"] == "18"
        assert env["T3_AIS_BUDGET_PATCH"] == "1"
        assert env["T3_AIS_EXEC_TIMEOUT"] == "900"
        assert env["T3D1_WALL_SECONDS"] == "9000"
        assert env["ABLATION_PHASE"] == "tranche3-d1"
        assert env["ABLATION_SYSTEM"] == "ai_scientist_v2"


def test_no_ablation_task_id_and_no_skip_flags() -> None:
    for job in load_jobs():
        env = env_map(job)
        assert "ABLATION_TASK_ID" not in env, f"{job['jobName']} sets ABLATION_TASK_ID"
        command = " ".join(job["containerOverrides"]["command"])
        assert "skip_writeup" not in command
        assert "skip_review" not in command
        assert "worker_native_d1.sh" in command
        assert "tranche3_worker_patch_v4b_d1_20260710.tar.gz" in command


def test_container_overrides_size_and_batch_shape() -> None:
    for job in load_jobs():
        size = len(json.dumps(job["containerOverrides"]))
        assert size <= 8000, f"{job['jobName']}: containerOverrides {size} chars"
        assert job["jobQueue"] == "mlevolve-ai-scientist-v2-ablation-queue-spot24"
        assert job["jobDefinition"] == "mlevolve-ai-scientist-v2-ablation-worker"
        assert job["timeout"] == {"attemptDurationSeconds": 10800}
        assert job["retryStrategy"] == {
            "attempts": 2,
            "evaluateOnExit": [
                {"action": "RETRY", "onStatusReason": "Host EC2*"},
                {"action": "EXIT", "onReason": "*"},
            ],
        }


def test_build_summary() -> None:
    summary = json.loads(SUMMARY_PATH.read_text())
    assert summary["jobs"] == 18
    assert summary["status"] == "BUILT_NOT_SUBMITTED"
    assert len(summary["ideas"]) == 3
    assert summary["no_ablation_task_id"] is True


def test_worker_native_d1_contract() -> None:
    subprocess.run(["bash", "-n", str(WORKER_SH)], check=True)
    text = WORKER_SH.read_text()
    code_only = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    assert "ABLATION_TASK_ID must not be set" in text
    assert "--skip_writeup" not in code_only
    assert "--skip_review" not in code_only
    assert "launch_scientist_bfts.py" in text
    assert "worker_ais_budget_patch.py" in text
    assert "review_text.txt" in text  # documents where perform_review writes


def test_collect_d1_reviews_local_fixture() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "tranche3-d1"
        # Completed run with paper + review.
        run_a = root / "t3d1-full-compreg-a1"
        exp_a = run_a / "ai_scientist_experiments/2026-07-11_00-00-00_compositional_regularization_nn_attempt_1"
        exp_a.mkdir(parents=True)
        (run_a / "run_status.json").write_text(json.dumps({
            "run_id": "t3d1-full-compreg-a1",
            "arm": "full_pipeline",
            "idea_slug": "compreg",
            "attempt_id": 1,
            "completed": True,
            "exit_status": 0,
        }))
        (exp_a / "review_text.txt").write_text(json.dumps({"Overall": 4, "Decision": "Reject"}))
        (exp_a / "template_reflection_final.pdf").write_bytes(b"%PDF-1.5 fake")
        # Failed run: no paper, no review.
        run_b = root / "t3d1-ddio-pestdetect-a2"
        (run_b / "ai_scientist_experiments").mkdir(parents=True)
        (run_b / "run_status.json").write_text(json.dumps({
            "run_id": "t3d1-ddio-pestdetect-a2",
            "arm": "draft_debug_improve_only",
            "idea_slug": "pestdetect",
            "attempt_id": 2,
            "completed": False,
            "exit_status": 124,
        }))

        out_csv = Path(tmp) / "summary.csv"
        subprocess.run(
            [sys.executable, str(COLLECTOR), "--root", str(root), "--out", str(out_csv)],
            check=True,
        )
        rows = {r["run_id"]: r for r in csv.DictReader(out_csv.open())}
        assert set(rows) == {"t3d1-full-compreg-a1", "t3d1-ddio-pestdetect-a2"}
        good = rows["t3d1-full-compreg-a1"]
        assert good["idea"] == "compreg"
        assert good["arm"] == "full_pipeline"
        assert good["attempt"] == "1"
        assert good["completed"] == "True"
        assert good["review_score"] == "4.0"
        assert good["paper_pdf_present"] == "True"
        bad = rows["t3d1-ddio-pestdetect-a2"]
        assert bad["completed"] == "False"
        assert bad["review_score"] == ""
        assert bad["paper_pdf_present"] == "False"


def main() -> int:
    tests = [
        test_job_count_and_matrix,
        test_env_correctness_per_arm,
        test_no_ablation_task_id_and_no_skip_flags,
        test_container_overrides_size_and_batch_shape,
        test_build_summary,
        test_worker_native_d1_contract,
        test_collect_d1_reviews_local_fixture,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
