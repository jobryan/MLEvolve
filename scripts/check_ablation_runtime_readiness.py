#!/usr/bin/env python3
"""Check whether the current host can run ablation experiments."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


ROOT = Path(__file__).resolve().parents[1]

CheckKind = Literal["module", "command", "env", "path"]


@dataclass(frozen=True)
class Check:
    name: str
    kind: CheckKind
    target: str
    required_for: tuple[str, ...]
    note: str = ""


CHECKS = [
    Check("python3", "command", "python3", ("mlevolve_smoke", "ais_v2_smoke", "screening")),
    Check("docker", "command", "docker", ("shared_mle_bench", "screening"), "Used by MLE-bench task execution."),
    Check("mlebench_cli", "command", "mlebench", ("shared_mle_bench", "screening")),
    Check("kaggle_cli", "command", "kaggle", ("shared_mle_bench_prepare", "screening")),
    Check("ai_scientist_v2_patch", "path", ".context/external/AI-Scientist-v2-ablation", ("ais_v2_smoke", "screening")),
    Check("torch", "module", "torch", ("mlevolve_smoke", "ais_v2_smoke", "screening")),
    Check("omegaconf", "module", "omegaconf", ("mlevolve_smoke", "ais_v2_smoke", "screening")),
    Check("rich", "module", "rich", ("mlevolve_smoke", "ais_v2_smoke", "screening")),
    Check("humanize", "module", "humanize", ("mlevolve_smoke", "ais_v2_smoke", "screening")),
    Check("dataclasses_json", "module", "dataclasses_json", ("mlevolve_smoke", "ais_v2_smoke", "screening")),
    Check("numpy", "module", "numpy", ("mlevolve_smoke", "ais_v2_smoke", "screening")),
    Check("pandas", "module", "pandas", ("mlevolve_smoke", "screening")),
    Check("coolname", "module", "coolname", ("mlevolve_smoke", "ais_v2_smoke", "screening")),
    Check("openai_module", "module", "openai", ("mlevolve_smoke", "ais_v2_smoke", "screening")),
    Check("google_genai", "module", "google.genai", ("mlevolve_smoke", "screening"), "Imported by the MLEvolve LLM package even when OpenAI-compatible models are selected."),
    Check("anthropic_module", "module", "anthropic", ("ais_v2_smoke", "screening")),
    Check("transformers", "module", "transformers", ("ais_v2_smoke", "screening")),
    Check("datasets", "module", "datasets", ("ais_v2_smoke", "screening")),
    Check("tiktoken", "module", "tiktoken", ("ais_v2_smoke", "screening")),
    Check("wandb", "module", "wandb", ("ais_v2_smoke", "screening")),
    Check("funcy", "module", "funcy", ("ais_v2_smoke", "screening")),
    Check("black", "module", "black", ("ais_v2_smoke", "screening")),
    Check("genson", "module", "genson", ("ais_v2_smoke", "screening")),
    Check("shutup", "module", "shutup", ("ais_v2_smoke", "screening")),
    Check("igraph", "module", "igraph", ("ais_v2_smoke", "screening"), "Python package from python-igraph."),
    Check("jsonschema", "module", "jsonschema", ("ais_v2_smoke", "screening")),
    Check("boto3", "module", "boto3", ("ais_v2_smoke", "screening")),
    Check("botocore", "module", "botocore", ("ais_v2_smoke", "screening")),
    Check("mlebench_module", "module", "mlebench", ("shared_mle_bench", "screening")),
    Check("OPENAI_API_KEY", "env", "OPENAI_API_KEY", ("ais_v2_smoke", "screening")),
    Check("KAGGLE_USERNAME", "env", "KAGGLE_USERNAME", ("shared_mle_bench_prepare", "screening")),
    Check("KAGGLE_KEY", "env", "KAGGLE_KEY", ("shared_mle_bench_prepare", "screening")),
]


def check_one(check: Check) -> dict:
    if check.kind == "module":
        try:
            ok = importlib.util.find_spec(check.target) is not None
        except ModuleNotFoundError:
            ok = False
    elif check.kind == "command":
        ok = shutil.which(check.target) is not None
    elif check.kind == "env":
        ok = bool(os.environ.get(check.target))
    elif check.kind == "path":
        ok = (ROOT / check.target).exists()
    else:
        raise ValueError(f"unknown check kind {check.kind!r}")

    return {
        "name": check.name,
        "kind": check.kind,
        "target": check.target if check.kind != "env" else check.name,
        "status": "pass" if ok else "fail",
        "required_for": list(check.required_for),
        "note": check.note,
    }


def summarize(results: list[dict]) -> dict:
    groups: dict[str, dict] = {}
    for result in results:
        for group in result["required_for"]:
            state = groups.setdefault(group, {"status": "pass", "missing": []})
            if result["status"] != "pass":
                state["status"] = "blocked"
                state["missing"].append(result["name"])
    return groups


def write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# Ablation Runtime Readiness",
        "",
        f"Execution environment: `{payload['environment_label']}`.",
        "",
        f"Overall status: `{payload['overall_status']}`.",
        "",
        "This report checks the current host only. If experiments run on AWS or another remote worker, run this script on that worker; local failures only block local execution.",
        "",
        "This report only records whether credentials are present. It never prints credential values.",
        "",
        "## Requirement Groups",
        "",
        "| Group | Status | Missing Checks |",
        "| --- | --- | --- |",
    ]
    for group, summary in sorted(payload["groups"].items()):
        missing = ", ".join(summary["missing"]) if summary["missing"] else ""
        lines.append(f"| {group} | {summary['status']} | {missing} |")

    lines.extend(
        [
            "",
            "## Individual Checks",
            "",
            "| Check | Kind | Status | Required For | Note |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for result in payload["checks"]:
        lines.append(
            f"| {result['name']} | {result['kind']} | {result['status']} | "
            f"{', '.join(result['required_for'])} | {result['note']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", default=Path(".context/ablation/runtime_readiness.json"), type=Path)
    parser.add_argument("--output-md", default=Path(".context/ablation/runtime_readiness.md"), type=Path)
    parser.add_argument(
        "--environment-label",
        default="current-host",
        help="Human-readable label for the host/environment being checked, e.g. aws-gpu-runner.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = [check_one(check) for check in CHECKS]
    groups = summarize(results)
    overall = "ready" if all(group["status"] == "pass" for group in groups.values()) else "blocked"
    payload = {
        "environment_label": args.environment_label,
        "overall_status": overall,
        "groups": groups,
        "checks": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.output_md, payload)
    print(f"runtime readiness: {overall}")
    print(f"json={args.output_json}")
    print(f"markdown={args.output_md}")
    return 0 if overall == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
