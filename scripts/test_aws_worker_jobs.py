#!/usr/bin/env python3
"""Tests for AWS worker manifest dispatch and job-spec rendering."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DISPATCH = ROOT / "scripts/run_ablation_manifest.py"
BUILD_JOBS = ROOT / "scripts/build_aws_ablation_jobs.py"
sys.path.insert(0, str(ROOT / "scripts"))
from build_aws_ablation_jobs import (  # noqa: E402
    ai_scientist_stage_budget,
    model_tier_environment,
    worker_patch_command,
)


def base_manifest(system: str, variant_id: str) -> dict:
    config_path = (
        "configs/ablations/mlevolve/default_mcgs.json"
        if system == "mlevolve"
        else "configs/ablations/ai_scientist_v2/stage_as_operator_default.json"
    )
    return {
        "schema_version": "1.0",
        "run_id": f"smoke-{system}-{variant_id}-spooky-seed-1",
        "system": system,
        "benchmark_track": "shared_mle_bench_lite",
        "task_id": "spooky-author-identification",
        "task_manifest_version": "mle-bench-lite-v1",
        "variant_registry_version": "ablation-variants-v1",
        "variant_config_sha1": "test",
        "variant_id": variant_id,
        "phase": "smoke",
        "seed": 1,
        "status": "planned",
        "task": {
            "id": "spooky-author-identification",
            "resource_class": "small_cpu",
            "metric_name": "multi-class-log-loss",
            "metric_direction": "minimize",
            "expected_submission_path": "/home/submission/submission.csv",
            "sample_submission_path": "spooky-author-identification/prepared/public/sample_submission.csv",
        },
        "variant": {
            "config_path": config_path,
            "cli_flags": ["--skip_writeup", "--skip_review"],
        },
        "config_overrides": {
            "agent.search.selection_mode": "mcgs" if system == "mlevolve" else "default_bfts"
        },
        "budget": {
            "wall_time_seconds": 14400,
            "max_nodes": 10,
            "max_cost_usd": 50.0,
        },
        "resource_policy": {
            "resource_class": "small_cpu",
            "matched_shared_benchmark": True,
        },
        "grader": {
            "name": "mle-bench",
            "version": "test",
            "command": "mlebench grade",
        },
        "artifacts": {
            "manifest_path": f".context/ablation/run_manifests/smoke-{system}-{variant_id}.json",
            "output_dir": f".context/ablation/runs/smoke-{system}-{variant_id}",
        },
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_manifest_dispatch_dry_run() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        mle_manifest = tmp / "mlevolve.json"
        ais_manifest = tmp / "ais.json"
        write_json(mle_manifest, base_manifest("mlevolve", "default_mcgs"))
        write_json(ais_manifest, base_manifest("ai_scientist_v2", "stage_as_operator_default"))

        mle_result = run_command([sys.executable, str(DISPATCH), "--manifest", str(mle_manifest), "--dry-run"])
        assert mle_result.returncode == 0, mle_result.stderr
        mle_spec = json.loads(mle_result.stdout)
        assert mle_spec["system"] == "mlevolve"
        assert "run.py" in mle_spec["runner"]["command"]
        assert "ablation.enabled=true" in mle_spec["runner"]["command"]
        assert "agent.steps=10" in mle_spec["runner"]["command"]
        assert "agent.code.model=${oc.env:MLEVOLVE_CODE_MODEL,${oc.env:MLEVOLVE_STRONG_CODE_MODEL,gpt-4.1}}" in mle_spec["runner"]["command"]
        assert "agent.feedback.model=${oc.env:MLEVOLVE_FEEDBACK_MODEL,${oc.env:MLEVOLVE_STRONG_FEEDBACK_MODEL,gpt-4.1}}" in mle_spec["runner"]["command"]
        assert "agent.code.api_key=${oc.env:MLEVOLVE_CODE_API_KEY,${oc.env:OPENAI_API_KEY,}}" in mle_spec["runner"]["command"]
        assert "agent.feedback.api_key=${oc.env:MLEVOLVE_FEEDBACK_API_KEY,${oc.env:OPENAI_API_KEY,}}" in mle_spec["runner"]["command"]

        ais_result = run_command([sys.executable, str(DISPATCH), "--manifest", str(ais_manifest), "--dry-run"])
        assert ais_result.returncode == 0, ais_result.stderr
        ais_spec = json.loads(ais_result.stdout)
        assert ais_spec["system"] == "ai_scientist_v2"
        assert "ablation_adapter/build_variant_command.py" in " ".join(ais_spec["runner"]["command"])
        assert "--ideas" in ais_spec["runner"]["command"]
        assert ais_spec["runner"]["env"]["ABLATION_TASK_ID"] == "spooky-author-identification"
        assert ais_spec["runner"]["env"]["ABLATION_SHARED_BENCHMARK"] == "1"
        assert ais_spec["runner"]["env"]["ABLATION_AI_SCIENTIST_IDEA_PATH"].endswith(
            "mle_bench_spooky-author-identification_seed_1_implementation.json"
        )
        assert ais_spec["runner"]["env"]["ABLATION_SAMPLE_SUBMISSION_PATH"].endswith(
            "spooky-author-identification/prepared/public/sample_submission.csv"
        )
        assert ais_spec["runner"]["env"]["ABLATION_METRIC_NAME"] == "multi-class-log-loss"
        assert ais_spec["runner"]["env"]["ABLATION_METRIC_DIRECTION"] == "minimize"
        assert ais_spec["runner"]["env"]["ABLATION_SKIP_AI_PLOTS"] == "1"


def test_aws_job_spec_rendering() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        manifest_path = tmp / "mlevolve.json"
        ais_manifest_path = tmp / "ais.json"
        output_jsonl = tmp / "jobs.jsonl"
        write_json(manifest_path, base_manifest("mlevolve", "default_mcgs"))
        write_json(ais_manifest_path, base_manifest("ai_scientist_v2", "stage_as_operator_default"))

        result = run_command(
            [
                sys.executable,
                str(BUILD_JOBS),
                "--manifest",
                str(manifest_path),
                "--manifest",
                str(ais_manifest_path),
                "--job-queue",
                "ablation-queue",
                "--job-definition",
                "ablation-jobdef",
                "--manifest-base-uri",
                "s3://example-bucket/manifests",
                "--output-jsonl",
                str(output_jsonl),
            ]
        )
        assert result.returncode == 0, result.stderr
        rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 2
        spec = rows[0]
        assert spec["jobQueue"] == "ablation-queue"
        assert spec["jobDefinition"] == "ablation-jobdef"
        assert spec["timeout"]["attemptDurationSeconds"] == 14700
        command = spec["containerOverrides"]["command"]
        assert command[:2] == ["bash", "-lc"]
        assert "mlebench prepare -c \"$TASK_ID\" --data-dir \"$MLEBENCH_DATASET_DIR\"" in command[2]
        assert "python3 scripts/run_ablation_manifest.py --manifest \"$MANIFEST_REF\"" in command[2]
        assert "python3 scripts/grade_ablation_submissions.py --output-dir \"$OUTPUT_DIR\"" in command[2]
        assert "|| true" in command[2]
        assert "ARTIFACT_DEST=\"${ABLATION_ARTIFACT_ROOT%/}/${PHASE}/${RUN_ID}/\"" in command[2]
        assert "boto3.client" in command[2]
        assert "upload_file" in command[2]
        # ARTIFACT_DEST must be computed before the run so the checkpoint loop can use it.
        assert command[2].index("ARTIFACT_DEST=\"${ABLATION_ARTIFACT_ROOT%/}/${PHASE}/${RUN_ID}/\"") < command[2].index(
            "python3 scripts/run_ablation_manifest.py"
        )
        # Periodic checkpoint sync keeps partial artifacts durable if the job hits the wall timeout.
        assert "CHECKPOINT_PID=$!" in command[2]
        assert "while true; do sleep 300;" in command[2]
        assert "kill \"$CHECKPOINT_PID\"" in command[2]
        # The checkpoint loop starts before the run and is killed before the final sync.
        assert command[2].index("CHECKPOINT_PID=$!") < command[2].index("python3 scripts/run_ablation_manifest.py")
        assert command[2].index("kill \"$CHECKPOINT_PID\"") < command[2].index("SYNC_STATUS=0;")
        # Exit-code semantics are unchanged: final sync status still decides when RUN_STATUS==0.
        assert "if [ \"$RUN_STATUS\" -ne 0 ]; then exit \"$RUN_STATUS\"; fi" in command[2]
        assert "exit \"$SYNC_STATUS\"" in command[2]
        assert "MANIFEST_REF=s3://example-bucket/manifests/" in command[2]
        resources = {item["type"]: item["value"] for item in spec["containerOverrides"]["resourceRequirements"]}
        assert resources["VCPU"] == "8"
        assert resources["MEMORY"] == "32768"
        assert "GPU" not in resources
        env = {item["name"]: item["value"] for item in spec["containerOverrides"]["environment"]}
        assert env["MLEVOLVE_NO_GPU"] == "1"
        assert env["MLEVOLVE_OFFLINE"] == "1"
        assert env["CUDA_VISIBLE_DEVICES"] == "-1"
        # Anchor jobs must never inherit model-tier routing env vars.
        assert "MLEVOLVE_STRONG_CODE_MODEL" not in env

        ais_command = rows[1]["containerOverrides"]["command"]
        assert ais_command[:2] == ["bash", "-lc"]
        assert "bfts_config.yaml" in ais_command[2]
        assert "model: gpt-4.1" in ais_command[2]
        assert "generate_report: False" in ais_command[2]
        assert "if not args.skip_writeup" in ais_command[2]
        assert "keywords = []" in ais_command[2]
        assert "SYSTEM=ai_scientist_v2" in ais_command[2]
        assert "AI_EXPERIMENTS_DIR=\".context/external/AI-Scientist-v2-ablation/experiments\"" in ais_command[2]
        assert "ai_scientist_experiments" in ais_command[2]
        assert "shutil.copytree" in ais_command[2]


def test_model_tier_environment_only_on_routing_variants() -> None:
    assert model_tier_environment("default_mcgs") == []
    assert model_tier_environment("no_memory") == []
    all_strong = {item["name"]: item["value"] for item in model_tier_environment("all_strong")}
    assert all_strong["MLEVOLVE_STRONG_CODE_MODEL"] == "gpt-5.5"
    assert all_strong["MLEVOLVE_STRONG_FEEDBACK_MODEL"] == "gpt-5.5"
    mixed = {
        item["name"]: item["value"]
        for item in model_tier_environment("strong_code_cheap_feedback")
    }
    assert mixed["MLEVOLVE_STRONG_CODE_MODEL"] == "gpt-5.5"
    assert mixed["MLEVOLVE_CHEAP_FEEDBACK_MODEL"] == "gpt-4.1-mini"
    # The strong tier must differ from the gpt-4.1 anchor or the variant is a placebo.
    assert all_strong["MLEVOLVE_STRONG_CODE_MODEL"] != "gpt-4.1"


def test_ai_scientist_stage_budget_prioritizes_initial_implementations() -> None:
    assert ai_scientist_stage_budget(1) == {
        "stage1_max_iters": 1,
        "stage2_max_iters": 0,
        "stage3_max_iters": 0,
        "stage4_max_iters": 0,
        "steps": 1,
    }
    assert ai_scientist_stage_budget(2) == {
        "stage1_max_iters": 2,
        "stage2_max_iters": 0,
        "stage3_max_iters": 0,
        "stage4_max_iters": 0,
        "steps": 2,
    }
    assert ai_scientist_stage_budget(5) == {
        "stage1_max_iters": 3,
        "stage2_max_iters": 1,
        "stage3_max_iters": 1,
        "stage4_max_iters": 0,
        "steps": 5,
    }
    # Screening budgets: stage 1 gets half the nodes so stage 4 cannot dominate.
    assert ai_scientist_stage_budget(10) == {
        "stage1_max_iters": 5,
        "stage2_max_iters": 1,
        "stage3_max_iters": 1,
        "stage4_max_iters": 3,
        "steps": 10,
    }


def test_worker_patch_command_downloads_s3_tarball() -> None:
    command = worker_patch_command({"worker_patch_uri": "s3://bucket/patches/contract.tar.gz"})
    assert "download_file" in command
    assert "tar.extractall" in command
    assert "unsafe patch path" in command
    assert "s3://bucket/patches/contract.tar.gz" in command


if __name__ == "__main__":
    test_manifest_dispatch_dry_run()
    test_aws_job_spec_rendering()
    test_model_tier_environment_only_on_routing_variants()
    test_ai_scientist_stage_budget_prioritizes_initial_implementations()
    test_worker_patch_command_downloads_s3_tarball()
    print("aws worker job tests passed")
