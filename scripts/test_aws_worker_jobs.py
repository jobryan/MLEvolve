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
    artifact_sync_timeout_buffer_seconds,
    ai_scientist_budget_environment,
    ai_scientist_stage_budget,
    build_job_spec,
    model_tier_environment,
    worker_patch_command,
)

WORKER_RUN_SH = ROOT / "scripts/worker_run.sh"


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
        ais_manifest = base_manifest("ai_scientist_v2", "stage_as_operator_default")
        ais_manifest["budget"]["max_debug_attempts"] = 2
        ais_manifest["runtime_controls"] = {
            "ai_scientist_budget_patch": True,
            "ais_exec_timeout": 900,
            "worker_patch_uri": "s3://example-bucket/patches/tranche3_worker_patch_v4b_test.tar.gz",
        }
        write_json(ais_manifest_path, ais_manifest)

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
        # The command is now a short bootstrap: export params, then hand off to
        # worker_run.sh (which arrives via the worker patch tarball).
        assert command[2].startswith("set -euo pipefail; export ")
        assert "SYSTEM=mlevolve" in command[2]
        assert "TASK_ID=spooky-author-identification" in command[2]
        assert "PHASE=smoke" in command[2]
        assert "RUN_ID=smoke-mlevolve-default_mcgs-spooky-seed-1" in command[2]
        assert "OUTPUT_DIR=.context/ablation/runs/smoke-mlevolve-default_mcgs" in command[2]
        assert "MANIFEST_REF=s3://example-bucket/manifests/" in command[2]
        assert command[2].endswith("bash scripts/worker_run.sh")
        # No mlevolve job should carry the AIS budget-patch env.
        assert "T3_AIS_" not in command[2]
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

        ais_spec = rows[1]
        ais_command = ais_spec["containerOverrides"]["command"]
        assert ais_command[:2] == ["bash", "-lc"]
        assert "SYSTEM=ai_scientist_v2" in ais_command[2]
        # AIS budget-patch parameters ride env vars consumed by worker_run.sh.
        assert "T3_AIS_BUDGET_PATCH=1" in ais_command[2]
        # min(ais_exec_timeout=900, wall=14400) clamped to [60, 3600] -> 900.
        assert "T3_AIS_EXEC_TIMEOUT=900" in ais_command[2]
        assert "T3_AIS_STAGE1=5" in ais_command[2]
        assert "T3_AIS_STAGE2=1" in ais_command[2]
        assert "T3_AIS_STAGE3=1" in ais_command[2]
        assert "T3_AIS_STAGE4=3" in ais_command[2]
        assert "T3_AIS_STEPS=10" in ais_command[2]
        assert "T3_AIS_NUM_DRAFTS=2" in ais_command[2]
        assert "T3_AIS_DEBUG_DEPTH=2" in ais_command[2]
        # The patch bootstrap stays inline; everything else lives in worker_run.sh.
        assert "download_file" in ais_command[2]
        assert "tranche3_worker_patch_v4b_test.tar.gz" in ais_command[2]
        assert ais_command[2].endswith("bash scripts/worker_run.sh")
        # Regression guard: AWS Batch rejects containerOverrides above 8192 chars.
        assert len(json.dumps(ais_spec["containerOverrides"])) <= 8000
        assert len(json.dumps(spec["containerOverrides"])) <= 8000


def test_worker_run_sh_contract() -> None:
    # The run body shipped in the patch tarball must be valid bash.
    result = run_command(["bash", "-n", str(WORKER_RUN_SH)])
    assert result.returncode == 0, result.stderr

    script = WORKER_RUN_SH.read_text(encoding="utf-8")
    assert "mlebench prepare -c \"$TASK_ID\" --data-dir \"$MLEBENCH_DATASET_DIR\"" in script
    assert "python3 scripts/run_ablation_manifest.py --manifest \"$MANIFEST_REF\"" in script
    assert "python3 scripts/grade_ablation_submissions.py --output-dir \"$OUTPUT_DIR\"" in script
    assert "|| true" in script
    assert "ARTIFACT_DEST=\"${ABLATION_ARTIFACT_ROOT%/}/${PHASE}/${RUN_ID}/\"" in script
    assert "boto3" in script
    assert "upload_file" in script
    # ARTIFACT_DEST must be computed before the run so the checkpoint loop can use it.
    assert script.index("ARTIFACT_DEST=\"${ABLATION_ARTIFACT_ROOT%/}/${PHASE}/${RUN_ID}/\"") < script.index(
        "python3 scripts/run_ablation_manifest.py"
    )
    # Periodic checkpoint sync keeps partial artifacts durable if the job hits the wall timeout.
    assert "CHECKPOINT_PID=$!" in script
    assert "sleep 300" in script
    assert "kill \"$CHECKPOINT_PID\"" in script
    assert script.index("CHECKPOINT_PID=$!") < script.index("python3 scripts/run_ablation_manifest.py")
    assert script.index("kill \"$CHECKPOINT_PID\"") < script.index("SYNC_STATUS=0")
    # Checkpoint uploads must exclude copied input data and verbose logs.
    assert "workspace/input/" in script
    assert ".verbose.log" in script
    assert "mode == 'checkpoint'" in script
    # AIS budget patch is applied only for ai_scientist_v2 with the env latch set.
    assert "worker_ais_budget_patch.py" in script
    assert "T3_AIS_BUDGET_PATCH" in script
    # AIS artifact recovery and copy steps survive the move out of the inline command.
    assert "AI_EXPERIMENTS_DIR=\".context/external/AI-Scientist-v2-ablation/experiments\"" in script
    assert "ai_scientist_experiments" in script
    assert "recover_ai_scientist_submission.py" in script
    # Exit-code semantics are unchanged: final sync status decides when RUN_STATUS==0.
    assert "if [ \"$RUN_STATUS\" -ne 0 ]; then exit \"$RUN_STATUS\"; fi" in script
    assert script.rstrip().endswith("exit \"$SYNC_STATUS\"")


def test_worker_ais_budget_patch_applies_replacements() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        (tmp / "bfts_config.yaml").write_text(
            "model: anthropic.claude-3-5-sonnet-20241022-v2:0\n"
            "generate_report: True\n"
            "  timeout: 3600\n"
            "  num_workers: 4\n"
            "    num_seeds: 3\n"
            "    num_drafts: 3\n"
            "    max_debug_depth: 3\n"
            "    stage1_max_iters: 20\n"
            "    stage2_max_iters: 12\n"
            "    stage3_max_iters: 12\n"
            "    stage4_max_iters: 18\n"
            "  steps: 5\n",
            encoding="utf-8",
        )
        (tmp / "launch_scientist_bfts.py").write_text(
            "    aggregate_plots(base_folder=idea_dir, model=args.model_agg_plots)\n\n"
            '    shutil.rmtree(osp.join(idea_dir, "experiment_results"))\n'
            '    keywords = ["python", "torch", "mp", "bfts", "experiment"]\n',
            encoding="utf-8",
        )
        result = run_command(
            [
                sys.executable,
                str(ROOT / "scripts/worker_ais_budget_patch.py"),
                "--root",
                str(tmp),
                "--exec-timeout",
                "900",
                "--stage1",
                "5",
                "--stage2",
                "1",
                "--stage3",
                "1",
                "--stage4",
                "3",
                "--steps",
                "10",
                "--num-drafts",
                "2",
                "--debug-depth",
                "2",
            ]
        )
        assert result.returncode == 0, result.stderr
        config = (tmp / "bfts_config.yaml").read_text(encoding="utf-8")
        assert "model: gpt-4.1" in config
        assert "generate_report: False" in config
        assert "  timeout: 900" in config
        assert "  num_workers: 1" in config
        assert "    num_drafts: 2" in config
        assert "    max_debug_depth: 2" in config
        assert "    stage1_max_iters: 5" in config
        assert "    stage4_max_iters: 3" in config
        assert "  steps: 10" in config
        launcher = (tmp / "launch_scientist_bfts.py").read_text(encoding="utf-8")
        assert "if not args.skip_writeup" in launcher
        assert "keywords = []" in launcher


def test_ai_scientist_budget_environment() -> None:
    # No budget patch flag -> no env at all.
    assert ai_scientist_budget_environment({"wall_time_seconds": 3000}, {}) == {}
    # Explicit override rides min(override, wall) with the [60, 3600] clamp.
    env = ai_scientist_budget_environment(
        {"wall_time_seconds": 3000, "max_nodes": 10, "max_debug_attempts": 2},
        {"ai_scientist_budget_patch": True, "ais_exec_timeout": 900},
    )
    assert env["T3_AIS_BUDGET_PATCH"] == "1"
    assert env["T3_AIS_EXEC_TIMEOUT"] == "900"
    assert env["T3_AIS_STAGE1"] == "5"
    assert env["T3_AIS_STEPS"] == "10"
    assert env["T3_AIS_NUM_DRAFTS"] == "2"
    assert env["T3_AIS_DEBUG_DEPTH"] == "2"
    # Without an override the wall budget is clamped to <= 3600 as before.
    env = ai_scientist_budget_environment(
        {"wall_time_seconds": 14400, "max_nodes": 3},
        {"ai_scientist_budget_patch": True},
    )
    assert env["T3_AIS_EXEC_TIMEOUT"] == "3600"
    assert env["T3_AIS_NUM_DRAFTS"] == "1"
    assert env["T3_AIS_DEBUG_DEPTH"] == "1"


def test_timeout_buffer_can_be_increased_per_manifest() -> None:
    manifest = base_manifest("mlevolve", "default_mcgs")
    manifest["runtime_controls"] = {"batch_timeout_buffer_seconds": 1200}
    spec = build_job_spec(
        manifest,
        {
            "aws_batch_defaults": {
                "vcpus_by_resource_class": {"small_cpu": 8},
                "memory_mb_by_resource_class": {"small_cpu": 32768},
                "gpus_by_resource_class": {"small_cpu": 0},
            }
        },
        job_queue="ablation-queue",
        job_definition="ablation-jobdef",
        manifest_base_uri="s3://example-bucket/manifests",
        include_tags=False,
    )
    assert spec["timeout"]["attemptDurationSeconds"] == 15600
    assert artifact_sync_timeout_buffer_seconds({"batch_timeout_buffer_seconds": 60}) == 300
    assert artifact_sync_timeout_buffer_seconds({"batch_timeout_buffer_seconds": "bad"}) == 300


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
    inverse = {
        item["name"]: item["value"]
        for item in model_tier_environment("cheap_code_strong_feedback")
    }
    assert inverse == {
        "MLEVOLVE_CHEAP_CODE_MODEL": "gpt-4.1-mini",
        "MLEVOLVE_STRONG_FEEDBACK_MODEL": "gpt-5.5",
    }
    all_cheap = {
        item["name"]: item["value"] for item in model_tier_environment("all_cheap")
    }
    assert all_cheap == {
        "MLEVOLVE_CHEAP_CODE_MODEL": "gpt-4.1-mini",
        "MLEVOLVE_CHEAP_FEEDBACK_MODEL": "gpt-4.1-mini",
    }
    # No tier may sit on the gpt-4.1 anchor or the variant is a placebo.
    assert all_strong["MLEVOLVE_STRONG_CODE_MODEL"] != "gpt-4.1"
    assert all_cheap["MLEVOLVE_CHEAP_CODE_MODEL"] != "gpt-4.1"
    assert inverse["MLEVOLVE_CHEAP_CODE_MODEL"] != "gpt-4.1"


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
    test_worker_run_sh_contract()
    test_worker_ais_budget_patch_applies_replacements()
    test_ai_scientist_budget_environment()
    test_timeout_buffer_can_be_increased_per_manifest()
    test_model_tier_environment_only_on_routing_variants()
    test_ai_scientist_stage_budget_prioritizes_initial_implementations()
    test_worker_patch_command_downloads_s3_tarball()
    print("aws worker job tests passed")
