#!/usr/bin/env python3
"""Apply the tranche-3 AI Scientist v2 runtime/budget patch on a worker.

This used to be an inline ``python3 -c`` one-liner emitted by
``build_aws_ablation_jobs.worker_command()``; it moved into this file (shipped
via the worker patch tarball and invoked from ``scripts/worker_run.sh``) so the
AWS Batch containerOverrides JSON stays below the 8192-character limit.

The string replacements are intentionally identical to the previous inline
patch: they rewrite ``bfts_config.yaml`` (model, report generation, exec
timeout, stage/node budgets) and ``launch_scientist_bfts.py`` (skip plot
aggregation with ``--skip_writeup``, keep experiment_results, disable the
process-name kill keywords).
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_AIS_ROOT = ".context/external/AI-Scientist-v2-ablation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_AIS_ROOT, help="AI-Scientist-v2 checkout to patch.")
    parser.add_argument("--exec-timeout", type=int, default=None, help="Per-node exec timeout (seconds).")
    parser.add_argument("--stage1", type=int, default=None, help="stage1_max_iters")
    parser.add_argument("--stage2", type=int, default=None, help="stage2_max_iters")
    parser.add_argument("--stage3", type=int, default=None, help="stage3_max_iters")
    parser.add_argument("--stage4", type=int, default=None, help="stage4_max_iters")
    parser.add_argument("--steps", type=int, default=None, help="agent steps (total nodes)")
    parser.add_argument("--num-drafts", type=int, default=1, help="num_drafts for the stage-budget block")
    parser.add_argument("--debug-depth", type=int, default=1, help="max_debug_depth for the stage-budget block")
    return parser.parse_args()


def apply_patch(args: argparse.Namespace) -> None:
    root = Path(args.root)

    config_path = root / "bfts_config.yaml"
    text = config_path.read_text()
    text = text.replace("model: anthropic.claude-3-5-sonnet-20241022-v2:0", "model: gpt-4.1")
    text = text.replace("generate_report: True", "generate_report: False")
    if args.exec_timeout is not None:
        text = text.replace("  timeout: 3600", f"  timeout: {args.exec_timeout}")
    stage_values = (args.stage1, args.stage2, args.stage3, args.stage4, args.steps)
    if all(value is not None for value in stage_values):
        text = text.replace("  num_workers: 4", "  num_workers: 1")
        text = text.replace("    num_seeds: 3", "    num_seeds: 1")
        text = text.replace("    num_drafts: 3", f"    num_drafts: {args.num_drafts}")
        text = text.replace("    max_debug_depth: 3", f"    max_debug_depth: {args.debug_depth}")
        text = text.replace("    max_tokens: 12000", "    max_tokens: 6000")
        text = text.replace("    max_tokens: 8192", "    max_tokens: 4000")
        text = text.replace("    stage1_max_iters: 20", f"    stage1_max_iters: {args.stage1}")
        text = text.replace("    stage2_max_iters: 12", f"    stage2_max_iters: {args.stage2}")
        text = text.replace("    stage3_max_iters: 12", f"    stage3_max_iters: {args.stage3}")
        text = text.replace("    stage4_max_iters: 18", f"    stage4_max_iters: {args.stage4}")
        text = text.replace("  steps: 5", f"  steps: {args.steps}")
    config_path.write_text(text)

    launcher_path = root / "launch_scientist_bfts.py"
    launcher = launcher_path.read_text()
    old = (
        "    aggregate_plots(base_folder=idea_dir, model=args.model_agg_plots)\n\n"
        '    shutil.rmtree(osp.join(idea_dir, "experiment_results"))'
    )
    new = (
        "    if not args.skip_writeup:\n"
        "        aggregate_plots(base_folder=idea_dir, model=args.model_agg_plots)\n\n"
        '    experiment_results_copy = osp.join(idea_dir, "experiment_results")\n'
        "    if os.path.exists(experiment_results_copy):\n"
        "        shutil.rmtree(experiment_results_copy)"
    )
    launcher = launcher.replace(old, new)
    launcher = launcher.replace(
        '    keywords = ["python", "torch", "mp", "bfts", "experiment"]',
        "    keywords = []",
    )
    launcher_path.write_text(launcher)


def main() -> int:
    apply_patch(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
