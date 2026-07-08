#!/usr/bin/env python3
"""Gate ablation AWS launches on job specs, budget cap, Kaggle access, and active jobs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from check_ablation_plan_budget import collect_manifests, summarize_plan
from check_kaggle_competition_downloads import (
    check_download,
    collect_competitions,
    credentials_from_args,
)
from submit_aws_ablation_jobs import iter_jsonl, job_summary


ACTIVE_STATUSES = ["SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING"]

# Anchor model used by the default variants (run_ablation_manifest.py env-resolver
# fallback). Model-routing jobs must route to something else, or the variant is
# a placebo replicate of the anchor — the exact tranche-1 failure.
ANCHOR_MODEL = "gpt-4.1"
MODEL_ROUTING_TIER_VARS = {
    "all_strong": ["MLEVOLVE_STRONG_CODE_MODEL", "MLEVOLVE_STRONG_FEEDBACK_MODEL"],
    "strong_code_cheap_feedback": ["MLEVOLVE_STRONG_CODE_MODEL", "MLEVOLVE_CHEAP_FEEDBACK_MODEL"],
}


def list_openai_models(api_key: str, timeout: float = 20.0) -> list[str] | None:
    """Model ids visible to the account, or None when the lookup fails."""
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None
    return [str(item.get("id", "")) for item in payload.get("data", [])]


def summarize_model_tiers(paths: list[Path], openai_api_key: str | None) -> dict[str, Any]:
    """Check model-routing job specs for the placebo-tier failure mode."""
    routing_jobs: list[dict[str, Any]] = []
    problems: list[str] = []
    tier_models: set[str] = set()
    for path in paths:
        for spec in iter_jsonl(path):
            variant_id = str(spec.get("parameters", {}).get("variant_id", ""))
            tier_vars = MODEL_ROUTING_TIER_VARS.get(variant_id)
            if not tier_vars:
                continue
            env = {
                item.get("name"): item.get("value")
                for item in spec.get("containerOverrides", {}).get("environment", [])
            }
            job_record = {"jobName": spec.get("jobName"), "variant_id": variant_id}
            routing_jobs.append(job_record)
            for var in tier_vars:
                value = env.get(var)
                if not value:
                    problems.append(f"{spec.get('jobName')}: {var} is unset")
                    continue
                tier_models.add(value)
                if var.startswith("MLEVOLVE_STRONG") and value == ANCHOR_MODEL:
                    problems.append(
                        f"{spec.get('jobName')}: {var}={value} equals the anchor model"
                    )
    summary: dict[str, Any] = {
        "routing_job_count": len(routing_jobs),
        "tier_models": sorted(tier_models),
        "problems": problems,
        "model_availability_checked": False,
    }
    if routing_jobs and not problems and tier_models and openai_api_key:
        available = list_openai_models(openai_api_key)
        if available is None:
            summary["warnings"] = ["openai_model_list_lookup_failed"]
        else:
            summary["model_availability_checked"] = True
            missing = sorted(model for model in tier_models if model not in available)
            if missing:
                summary["problems"] = problems = [
                    f"model not available on this OpenAI account: {model}" for model in missing
                ]
    summary["ok"] = not problems
    return summary


def list_active_jobs(job_queue: str, aws_cli: str, region: str) -> dict[str, list[dict[str, str]]]:
    summary: dict[str, list[dict[str, str]]] = {}
    for status in ACTIVE_STATUSES:
        command = [
            aws_cli,
            "batch",
            "list-jobs",
            "--region",
            region,
            "--job-queue",
            job_queue,
            "--job-status",
            status,
            "--output",
            "json",
        ]
        completed = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or f"list-jobs failed for status {status}")
        jobs = json.loads(completed.stdout).get("jobSummaryList", [])
        summary[status] = [
            {"jobId": str(job.get("jobId", "")), "jobName": str(job.get("jobName", ""))}
            for job in jobs
        ]
    return summary


def summarize_kaggle(args: argparse.Namespace) -> dict[str, Any]:
    if args.kaggle_summary_json:
        return json.loads(args.kaggle_summary_json.read_text(encoding="utf-8"))
    if args.skip_kaggle:
        return {"skipped": True, "all_ok": True, "checks": []}

    competitions = collect_competitions(args)
    if not competitions:
        return {"skipped": True, "all_ok": True, "checks": []}

    username, key = credentials_from_args(args)
    checks = [check_download(competition, username, key, timeout=args.timeout) for competition in competitions]
    return {
        "worker_kaggle_username": username,
        "checked_count": len(checks),
        "ok_count": sum(1 for check in checks if check.ok),
        "failed_count": sum(1 for check in checks if not check.ok),
        "all_ok": all(check.ok for check in checks),
        "checks": [check.__dict__ for check in checks],
    }


def summarize_jobs(paths: list[Path]) -> dict[str, Any]:
    specs: list[dict[str, Any]] = []
    for path in paths:
        specs.extend(iter_jsonl(path))
    if not specs:
        return {"skipped": True, "ready_for_submit": True, "placeholder_count": 0, "job_count": 0}
    return job_summary(specs)


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []

    jobs = summarize_jobs(args.jobs_jsonl)
    if jobs.get("placeholder_count", 0):
        blockers.append("job_specs_contain_placeholders")

    model_tiers = summarize_model_tiers(
        args.jobs_jsonl, openai_api_key=os.environ.get("OPENAI_API_KEY")
    )
    if not model_tiers["ok"]:
        blockers.append("model_routing_tier_missing_or_placebo")
    warnings.extend(model_tiers.get("warnings", []))

    if args.manifest_jsonl or args.manifest:
        budget = summarize_plan(
            collect_manifests(args.manifest, args.manifest_jsonl),
            spend_cap_usd=args.spend_cap_usd,
        )
        if args.spend_cap_usd is None and not args.allow_missing_spend_cap:
            blockers.append("missing_spend_cap_usd")
        if budget["over_cap"]:
            blockers.append("planned_budget_exceeds_spend_cap")
    else:
        budget = {"skipped": True, "spend_cap_usd": args.spend_cap_usd, "over_cap": False}
        warnings.append("no_manifests_provided_for_budget_check")
        if args.spend_cap_usd is None and not args.allow_missing_spend_cap:
            blockers.append("missing_spend_cap_usd")

    kaggle = summarize_kaggle(args)
    if not kaggle.get("all_ok", False):
        blockers.append("kaggle_download_preflight_failed")

    if args.skip_active_jobs or not args.job_queue:
        active_jobs = {"skipped": True, "active_count": 0, "by_status": {}}
    else:
        by_status = list_active_jobs(args.job_queue, args.aws_cli, args.region)
        active_count = sum(len(jobs_for_status) for jobs_for_status in by_status.values())
        active_jobs = {"skipped": False, "active_count": active_count, "by_status": by_status}
        if active_count and not args.allow_active_jobs:
            blockers.append("active_batch_jobs_present")

    return {
        "ready": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "jobs": jobs,
        "model_tiers": model_tiers,
        "budget": budget,
        "kaggle": kaggle,
        "active_jobs": active_jobs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-jsonl", action="append", type=Path, default=[], help="AWS Batch job specs JSONL.")
    parser.add_argument("--manifest", action="append", type=Path, default=[], help="Run manifest JSON.")
    parser.add_argument("--manifest-jsonl", action="append", type=Path, default=[], help="Run manifests JSONL.")
    parser.add_argument("--spend-cap-usd", type=float)
    parser.add_argument("--allow-missing-spend-cap", action="store_true")
    parser.add_argument("--kaggle-summary-json", type=Path, help="Use an existing Kaggle preflight artifact.")
    parser.add_argument("--skip-kaggle", action="store_true")
    parser.add_argument("--credential-source", choices=["aws-secrets", "env"], default="aws-secrets")
    parser.add_argument("--username-secret", default="mlevolve-ai-scientist-v2-ablation/kaggle/username")
    parser.add_argument("--key-secret", default="mlevolve-ai-scientist-v2-ablation/kaggle/key")
    parser.add_argument("--kaggle-username", default=None)
    parser.add_argument("--kaggle-key", default=None)
    parser.add_argument("--task-manifest", type=Path)
    parser.add_argument("--competition", action="append", default=[])
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--job-queue", help="AWS Batch queue to inspect for active jobs.")
    parser.add_argument("--allow-active-jobs", action="store_true")
    parser.add_argument("--skip-active-jobs", action="store_true")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--aws-cli", default="aws")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_summary(args)
    encoded = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if summary["ready"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"launch readiness check failed: {exc}", file=sys.stderr)
        raise
