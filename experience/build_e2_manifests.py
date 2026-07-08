"""Generate E2 (fold-swap experiment) run manifests in the phoenix ablation schema.

Emits one JSONL manifest per (task, seed) for a given arm, compatible with the
anchored harness's `scripts/run_ablation_manifest.py` worker entrypoint
(config_overrides are OmegaConf dotted keys). Job-spec rendering consumes
`runtime_controls` (worker patch, store URI, Batch attempt timeout).

Arms:
  A  experience off (anchor behavior; doubles as the experience corpus source)
  B  self-experience: fold_a tasks read the fold_b store and vice versa
  C  placebo stores, same fold-swap routing
  D  foreign store, stratified task subset

Usage (arm A example):
    python -m experience.build_e2_manifests --arm A --phase screening \
        --run-id-prefix e2e1-v1 --seeds 1,2,3 \
        --worker-patch-uri s3://.../patches/e2_worker_patch_YYYYMMDD.tar.gz \
        --worker-patch-sha256 <sha256> \
        --output .context/e2/manifests_arm_a.jsonl

Arms B/C/D additionally need --store-uri-fold-a/--store-uri-fold-b (S3 URIs of
the OPPOSITE-fold stores are chosen per task automatically) and the matching
--store-snapshot-fold-a/--store-snapshot-fold-b hashes.

Deliberate choices (see experience/E1_RUNBOOK.md):
  - --phase is REQUIRED (the anchored subset builder's silent `smoke` default
    caused a phase mislabel incident in phoenix).
  - Batch attempt timeout is emitted as wall_time + 900s so overruns cannot
    destroy artifacts.
  - Anchor identifiers (E1 anchor commit, base tarball sha, image digest, job
    definition) are embedded in every manifest.
  - The anchor's memory config is NOT modified (global memory silently disables
    on CPU workers in all arms alike; ingestion falls back to journal mining).
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPLIT = Path(__file__).resolve().parent / "mle_bench_lite_split.json"
DEFAULT_TASK_MANIFEST = REPO_ROOT / "configs" / "ablations" / "tasks" / "mle_bench_lite.json"

E1_ANCHOR = {
    "e1_anchor_commit": "602291e19134b3f3e42cc20afd999c86e333d023",
    "base_worker_patch": "tranche2_worker_patch_20260706_v3.tar.gz",
    "base_worker_patch_sha256": "7f01a9110b7fb2b1569ed62290892d221f42fba9b3b59cb4ebc56d2a166a3a3f",
    "worker_image_digest": "sha256:5c9f3e82ec427b325f5fff731c8b0def46f537528dd57a8a079d16c778625129",
    "job_definition": "mlevolve-ai-scientist-v2-ablation-worker:1",
}

WORKER_STORE_DIR = ".context/e2_store"

ARM_SPECS = {
    "A": {
        "variant_id": "exp_off",
        "summary": "Anchor MLEvolve with the cross-run experience layer disabled (control; corpus source).",
        "research_question": "Baseline: what does the anchor achieve on each task without cross-run experience?",
    },
    "B": {
        "variant_id": "exp_self",
        "summary": "Cross-run self-experience enabled; store built from arm-A runs of the opposite fold.",
        "research_question": "C1: does self-generated cross-run experience improve held-out task performance?",
    },
    "C": {
        "variant_id": "exp_placebo",
        "summary": "Placebo store (content-deranged, size/token-matched) with identical injection prompts.",
        "research_question": "C2: is the arm-B gain content-specific, or a prompt-token/scaffolding effect?",
    },
    "D": {
        "variant_id": "exp_foreign",
        "summary": "Foreign experience store (not self-generated) with identical injection prompts.",
        "research_question": "Does the system's OWN experience beat generic/foreign experience?",
    },
}

# Reduced profile per phoenix's t2x readout (2026-07-07): at 10-node/50-min CPU
# budgets MLEvolve validity was ~44% (0 valid Spooky rows); at 3-4 nodes/30-min
# it completes. E2 runs adopt the recovery-batch profile.
MICRO_CONTROLS = {
    "agent.initial_drafts": 1,
    "agent.search.num_drafts": 1,
    "agent.search.parallel_search_num": 1,
    "agent.search.max_debug_depth": 1,
}

MICRO_BUDGET = {
    "budget_policy_version": "e2e1-2026-07-07-budget-v2-reduced",
    "max_cost_usd": 1.5,
    "max_nodes": 4,
    "wall_time_seconds": 1800,
    "max_debug_attempts": 1,
    "max_input_tokens": 1500000,
    "max_output_tokens": 250000,
    "early_stop": {"catastrophic_valid_node_rate": 0.0, "min_nodes_before_check": 1},
}


def stable_suffix(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def build_manifest(
    arm: str,
    task: Dict[str, Any],
    seed: int,
    phase: str,
    prefix: str,
    worker_patch_uri: str,
    worker_patch_sha256: str,
    fold_of: Dict[str, str],
    folds: Dict[str, List[str]],
    store_uris: Dict[str, str],
    store_snapshots: Dict[str, str],
    store_tarball_shas: Dict[str, str],
    budget: Dict[str, Any],
    e2_commit: str,
) -> Dict[str, Any]:
    spec = ARM_SPECS[arm]
    task_id = task["id"]
    base = f"{prefix}-{phase}-mlevolve-{spec['variant_id']}-{task_id}-seed-{seed}"
    run_id = f"{base}-{stable_suffix(base)}"

    overrides: Dict[str, Any] = dict(MICRO_CONTROLS)
    overrides["agent.seed"] = seed
    overrides["agent.experience.enabled"] = arm != "A"

    runtime_controls: Dict[str, Any] = {
        "worker_patch_uri": worker_patch_uri,
        "worker_patch_sha256": worker_patch_sha256,
        "batch_attempt_timeout_seconds": budget["wall_time_seconds"] + 900,
    }

    if arm != "A":
        own_fold = fold_of[task_id]
        opposite = "fold_b" if own_fold == "fold_a" else "fold_a"
        overrides["agent.experience.store_dir"] = WORKER_STORE_DIR
        overrides["agent.experience.task_id"] = task_id
        overrides["agent.experience.excluded_task_ids"] = list(folds[own_fold])
        runtime_controls["store_uri"] = store_uris[opposite]
        runtime_controls["store_snapshot"] = store_snapshots[opposite]
        runtime_controls["store_tarball_sha256"] = store_tarball_shas.get(opposite, "")
        runtime_controls["store_fold"] = opposite

    return {
        "schema_version": "1.0",
        "benchmark_track": "shared_mle_bench_lite",
        "phase": phase,
        "status": "planned",
        "system": "mlevolve",
        "run_id": run_id,
        "seed": seed,
        # Flat keys consumed by scripts/run_ablation_manifest.py on the worker
        # (the nested task/variant blocks below are audit metadata; the worker
        # reads manifest["task_id"] / manifest["variant_id"] directly).
        "task_id": task_id,
        "variant_id": spec["variant_id"],
        "task": task,
        "variant": {
            "component_class": "experience_layer",
            "cli_flags": [],
            "config_path": None,
            "variant_id": spec["variant_id"],
            "summary": spec["summary"],
            "research_question": spec["research_question"],
        },
        "config_overrides": overrides,
        "budget": budget,
        "grader": {
            "name": "mle-bench",
            "version": "external_repo_snapshot",
            "command": task.get("grade_command", "mlebench grade"),
            "grade_sample_command": task.get("grade_sample_command", ""),
        },
        "resource_policy": {
            "matched_shared_benchmark": True,
            "resource_class": task.get("resource_class", ""),
        },
        "runtime_controls": runtime_controls,
        "anchor": {**E1_ANCHOR, "e2_branch_commit": e2_commit},
        "artifacts": {
            "manifest_path": f".context/e2/run_manifests/{run_id}.json",
            "output_dir": f".context/e2/runs/{run_id}",
        },
        "notes": f"E2 fold-swap experiment arm {arm} ({spec['variant_id']}); split v4 frozen 2026-07-02.",
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, choices=["A", "B", "C", "D"])
    parser.add_argument("--phase", required=True, help="Explicit phase (no silent default; see runbook)")
    parser.add_argument("--run-id-prefix", required=True)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--worker-patch-uri", required=True)
    parser.add_argument("--worker-patch-sha256", required=True)
    parser.add_argument("--e2-commit", default="", help="E2 branch commit for the anchor block (default: git HEAD)")
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--task-manifest", type=Path, default=DEFAULT_TASK_MANIFEST)
    parser.add_argument("--store-uri-fold-a", default="", help="S3 URI of the fold_a-built store (read by fold_b tasks)")
    parser.add_argument("--store-uri-fold-b", default="", help="S3 URI of the fold_b-built store (read by fold_a tasks)")
    parser.add_argument("--store-snapshot-fold-a", default="")
    parser.add_argument("--store-snapshot-fold-b", default="")
    parser.add_argument("--store-tarball-sha-fold-a", default="", help="sha256 of the packed fold_a store tarball")
    parser.add_argument("--store-tarball-sha-fold-b", default="", help="sha256 of the packed fold_b store tarball")
    parser.add_argument("--tasks", default="", help="Comma-separated task subset (e.g. arm D); default all non-canary")
    parser.add_argument("--include-canary", action="store_true", help="Also emit canary tasks (smoke use only)")
    parser.add_argument("--max-cost-usd", type=float, default=MICRO_BUDGET["max_cost_usd"])
    parser.add_argument("--wall-time-seconds", type=int, default=MICRO_BUDGET["wall_time_seconds"])
    parser.add_argument("--max-nodes", type=int, default=MICRO_BUDGET["max_nodes"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    split = json.loads(args.split.read_text())
    folds = {"fold_a": split["fold_a"], "fold_b": split["fold_b"]}
    fold_of = {t: f for f, tasks in folds.items() for t in tasks}
    task_manifest = json.loads(args.task_manifest.read_text())
    tasks_by_id = {t["id"]: t for t in task_manifest["tasks"]}

    if args.arm != "A":
        missing = [n for n in ("--store-uri-fold-a", "--store-uri-fold-b",
                               "--store-snapshot-fold-a", "--store-snapshot-fold-b")
                   if not getattr(args, n.lstrip("-").replace("-", "_"))]
        if missing:
            parser.error(f"arm {args.arm} requires {', '.join(missing)}")

    if args.tasks:
        task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
        unknown = [t for t in task_ids if t not in tasks_by_id]
        if unknown:
            parser.error(f"unknown task ids: {unknown}")
        non_canary_unknown = [t for t in task_ids if t not in fold_of and t not in split["canary"]]
        if non_canary_unknown:
            parser.error(f"task ids not in any fold: {non_canary_unknown}")
    else:
        task_ids = folds["fold_a"] + folds["fold_b"]
    if args.include_canary:
        task_ids = task_ids + [t for t in split["canary"] if t not in task_ids]
    canary_in_arm = [t for t in task_ids if t not in fold_of]
    if canary_in_arm and args.arm != "A":
        parser.error(f"canary tasks only run under arm A (no fold store exists for them): {canary_in_arm}")

    e2_commit = args.e2_commit
    if not e2_commit:
        import subprocess

        e2_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT,
        ).stdout.strip()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    budget = dict(MICRO_BUDGET)
    budget.update({
        "max_cost_usd": args.max_cost_usd,
        "wall_time_seconds": args.wall_time_seconds,
        "max_nodes": args.max_nodes,
    })
    store_uris = {"fold_a": args.store_uri_fold_a, "fold_b": args.store_uri_fold_b}
    store_snapshots = {"fold_a": args.store_snapshot_fold_a, "fold_b": args.store_snapshot_fold_b}
    store_tarball_shas = {"fold_a": args.store_tarball_sha_fold_a, "fold_b": args.store_tarball_sha_fold_b}

    manifests = [
        build_manifest(
            args.arm, tasks_by_id[task_id], seed, args.phase, args.run_id_prefix,
            args.worker_patch_uri, args.worker_patch_sha256,
            fold_of, folds, store_uris, store_snapshots, store_tarball_shas, budget, e2_commit,
        )
        for task_id in task_ids
        for seed in seeds
    ]

    total_cost = sum(m["budget"]["max_cost_usd"] for m in manifests)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for m in manifests:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"arm {args.arm}: wrote {len(manifests)} manifests to {args.output}")
    print(f"tasks={len(task_ids)} seeds={seeds} planned worst-case exposure=${total_cost:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
