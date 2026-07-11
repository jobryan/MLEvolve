#!/usr/bin/env python3
"""Build Tranche-3 Block D1 (native research track) AWS Batch job specs.

Block D1 runs AI Scientist v2 in its NATIVE mode: research idea in ->
experiments + paper writeup + LLM review out. No MLE-bench task, no Kaggle
data, no mlebench grading — so this deliberately bypasses the manifest
machinery (run_ablation_manifest / build_aws_ablation_jobs schemas) and
hand-constructs the Batch job JSONL.

Arms (writeup + review ENABLED for both — no skip flags):
  - full_pipeline: all four bfts stages (stage1=6, stage2=2, stage3=6,
    stage4=4, steps=18).
  - draft_debug_improve_only: stages 3/4 disabled via AIS_DISABLED_STAGES=3,4
    (stage1=12, stage2=6, stage3=0, stage4=0, steps=18 — budget parity).

Jobs never set ABLATION_TASK_ID: that env var activates the MLE-bench
contract audit inside the fork (_ablation_contract_enabled).

Outputs:
  .context/ablation/aws_jobs/t3d1_jobs_spot24.jsonl
  .context/ablation/aws_jobs/t3d1_build_summary.json
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_aws_ablation_jobs import worker_patch_command  # noqa: E402

JOB_QUEUE = "mlevolve-ai-scientist-v2-ablation-queue-spot24"
JOB_DEFINITION = "mlevolve-ai-scientist-v2-ablation-worker"
WORKER_PATCH_URI = (
    "s3://autoresearch-experiments-058264252788-us-east-1/"
    "mlevolve-ai-scientist-v2-ablation/patches/"
    "tranche3_worker_patch_v4b_d1_20260710.tar.gz"
)
PHASE = "tranche3-d1"
WALL_SECONDS = 9000
BATCH_TIMEOUT_SECONDS = 10800
ATTEMPTS = (1, 2, 3)
CONTAINER_OVERRIDES_LIMIT = 8000
EST_COST_PER_RUN_USD = (3.0, 5.0)  # writeup-heavy native runs

# Native (non-MLE-bench) idea JSONs shipped in the fork. Only 3 unique ideas
# exist under ai_scientist/ideas/ (the ...realworld.json file duplicates
# real_world_pest_detection), short of the 6 the design asked for.
IDEA_FILE = "ai_scientist/ideas/i_cant_believe_its_not_better.json"
IDEAS = [
    {"idea_idx": 0, "name": "compositional_regularization_nn", "slug": "compreg"},
    {"idea_idx": 1, "name": "interpretability_failure_modes", "slug": "interpfail"},
    {"idea_idx": 2, "name": "real_world_pest_detection", "slug": "pestdetect"},
]

ARMS = {
    "full_pipeline": {
        "slug": "full",
        "stage_env": {
            "T3_AIS_STAGE1": "6",
            "T3_AIS_STAGE2": "2",
            "T3_AIS_STAGE3": "6",
            "T3_AIS_STAGE4": "4",
        },
        "extra_env": {},
    },
    "draft_debug_improve_only": {
        "slug": "ddio",
        "stage_env": {
            "T3_AIS_STAGE1": "12",
            "T3_AIS_STAGE2": "6",
            "T3_AIS_STAGE3": "0",
            "T3_AIS_STAGE4": "0",
        },
        "extra_env": {"AIS_DISABLED_STAGES": "3,4"},
    },
}


def bootstrap_command(run_id: str, output_dir: str) -> list[str]:
    exports = {
        "SYSTEM": "ai_scientist_v2",
        "PHASE": PHASE,
        "RUN_ID": run_id,
        "OUTPUT_DIR": output_dir,
    }
    export_clause = "export " + " ".join(
        f"{name}={shlex.quote(str(value))}" for name, value in exports.items()
    )
    script = (
        "set -euo pipefail; "
        f"{export_clause}; "
        + worker_patch_command({"worker_patch_uri": WORKER_PATCH_URI})
        + "bash scripts/worker_native_d1.sh"
    )
    return ["bash", "-lc", script]


def build_job(arm: str, idea: dict, attempt: int) -> dict:
    arm_cfg = ARMS[arm]
    job_name = f"t3d1-{arm_cfg['slug']}-{idea['slug']}-a{attempt}"
    run_id = job_name
    output_dir = f".context/ablation/runs/{run_id}"

    env = {
        "ABLATION_RUN_ID": run_id,
        "ABLATION_SYSTEM": "ai_scientist_v2",
        "ABLATION_VARIANT_ID": arm,
        "ABLATION_PHASE": PHASE,
        "CUDA_VISIBLE_DEVICES": "-1",
        # bfts runtime/budget clamp (applied by worker_native_d1.sh).
        "T3_AIS_BUDGET_PATCH": "1",
        "T3_AIS_EXEC_TIMEOUT": "900",
        **arm_cfg["stage_env"],
        "T3_AIS_STEPS": "18",
        "T3_AIS_NUM_DRAFTS": "2",
        "T3_AIS_DEBUG_DEPTH": "3",
        **arm_cfg["extra_env"],
        # Native-run inputs.
        "T3D1_IDEA_PATH": IDEA_FILE,
        "T3D1_IDEA_IDX": str(idea["idea_idx"]),
        "T3D1_IDEA_SLUG": idea["slug"],
        "T3D1_ATTEMPT_ID": str(attempt),
        "T3D1_WALL_SECONDS": str(WALL_SECONDS),
    }
    assert "ABLATION_TASK_ID" not in env

    container_overrides = {
        "command": bootstrap_command(run_id, output_dir),
        "environment": [{"name": k, "value": v} for k, v in env.items()],
        "resourceRequirements": [
            {"type": "VCPU", "value": "8"},
            {"type": "MEMORY", "value": "32768"},
        ],
    }
    overrides_len = len(json.dumps(container_overrides))
    if overrides_len > CONTAINER_OVERRIDES_LIMIT:
        raise ValueError(
            f"{job_name}: containerOverrides JSON {overrides_len} chars "
            f"exceeds {CONTAINER_OVERRIDES_LIMIT}"
        )

    return {
        "jobName": job_name,
        "jobQueue": JOB_QUEUE,
        "jobDefinition": JOB_DEFINITION,
        "parameters": {
            "phase": PHASE,
            "run_id": run_id,
            "system": "ai_scientist_v2",
            "variant_id": arm,
            "idea": idea["name"],
            "attempt": str(attempt),
        },
        "containerOverrides": container_overrides,
        "retryStrategy": {
            "attempts": 2,
            "evaluateOnExit": [
                {"action": "RETRY", "onStatusReason": "Host EC2*"},
                {"action": "EXIT", "onReason": "*"},
            ],
        },
        "timeout": {"attemptDurationSeconds": BATCH_TIMEOUT_SECONDS},
    }


def main() -> int:
    jobs = [
        build_job(arm, idea, attempt)
        for arm in ARMS
        for idea in IDEAS
        for attempt in ATTEMPTS
    ]

    out_dir = ROOT / ".context/ablation/aws_jobs"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs_path = out_dir / "t3d1_jobs_spot24.jsonl"
    with jobs_path.open("w") as fh:
        for job in jobs:
            fh.write(json.dumps(job, sort_keys=True) + "\n")

    summary = {
        "generated_utc": "2026-07-10",
        "status": "BUILT_NOT_SUBMITTED",
        "block": "D1-native-research-track",
        "phase": PHASE,
        "worker_patch_uri": WORKER_PATCH_URI,
        "job_queue": JOB_QUEUE,
        "job_definition": JOB_DEFINITION,
        "worker_entrypoint": "scripts/worker_native_d1.sh",
        "arms": {
            arm: {**cfg["stage_env"], **cfg["extra_env"], "T3_AIS_STEPS": "18"}
            for arm, cfg in ARMS.items()
        },
        "idea_file": IDEA_FILE,
        "ideas": [
            {"idea_idx": i["idea_idx"], "name": i["name"], "slug": i["slug"]}
            for i in IDEAS
        ],
        "ideas_note": (
            "Design asked for 6 native idea JSONs; the fork ships only 3 unique "
            "native ideas (ai_scientist/ideas/i_cant_believe_its_not_better.json; "
            "the ...realworld.json file duplicates real_world_pest_detection). "
            "All other idea JSONs in the fork are MLE-bench-shaped and excluded."
        ),
        "attempts_per_idea_per_arm": len(ATTEMPTS),
        "jobs": len(jobs),
        "wall_seconds": WALL_SECONDS,
        "batch_timeout_seconds_per_attempt": BATCH_TIMEOUT_SECONDS,
        "est_cost_usd": {
            "per_run": list(EST_COST_PER_RUN_USD),
            "total_low": EST_COST_PER_RUN_USD[0] * len(jobs),
            "total_high": EST_COST_PER_RUN_USD[1] * len(jobs),
        },
        "outcome_variables": [
            "end_to_end_completion (run_status.json)",
            "llm_review_score (review_text.txt Overall)",
            "paper artifacts for later human rubric (PDF/tex)",
        ],
        "no_ablation_task_id": True,
    }
    summary_path = out_dir / "t3d1_build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"wrote {len(jobs)} jobs -> {jobs_path}")
    print(f"wrote summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
