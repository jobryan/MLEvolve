#!/usr/bin/env python3
"""Novelty mediation analysis over ablation nodes.jsonl exports (stdlib only).

Reads node records (schema: configs/ablations/schema/node.schema.json) and emits
per-run diversity summaries (mean nearest-neighbor distance, strategy-label
entropy) plus a Pearson-correlation table relating novelty_lambda_in_effect to
(i) diversity metrics and (ii) run best validation score.

Usage:
    python scripts/analyze_novelty_mediation.py nodes.jsonl [more.jsonl ...] \
        [--json-out report.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def iter_nodes(paths: list[Path]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"warning: {path}:{line_no}: bad JSON ({exc})", file=sys.stderr)
                    continue
                if isinstance(record, dict):
                    nodes.append(record)
    return nodes


def strategy_signature(strategy_labels: Any) -> str | None:
    """Collapse a node's strategy labels into one hashable signature."""
    if isinstance(strategy_labels, dict):
        return "|".join(f"{key}={value}" for key, value in sorted(strategy_labels.items()))
    if isinstance(strategy_labels, list):
        return "|".join(str(value) for value in sorted(strategy_labels))
    if strategy_labels is None:
        return None
    return str(strategy_labels)


def shannon_entropy(labels: list[str]) -> float | None:
    """Entropy (bits) of the label distribution; None when no labels."""
    if not labels:
        return None
    counts = Counter(labels)
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation by hand; None if undefined (n<2 or zero variance)."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    var_x = sum((x - mx) ** 2 for x in xs)
    var_y = sum((y - my) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    return cov / math.sqrt(var_x * var_y)


def _diversity(node: dict[str, Any]) -> dict[str, Any]:
    diversity = node.get("diversity")
    return diversity if isinstance(diversity, dict) else {}


def _best_validation_score(nodes: list[dict[str, Any]]) -> float | None:
    scores: list[float] = []
    minimize = False
    for node in nodes:
        value = node.get("validation_score")
        if isinstance(value, (int, float)):
            scores.append(float(value))
        if node.get("metric_direction") == "minimize":
            minimize = True
    if not scores:
        return None
    return min(scores) if minimize else max(scores)


def summarize_run(run_id: str, nodes: list[dict[str, Any]]) -> dict[str, Any]:
    distances = [
        float(_diversity(node).get("nearest_neighbor_distance"))
        for node in nodes
        if isinstance(_diversity(node).get("nearest_neighbor_distance"), (int, float))
    ]
    signatures = [
        signature
        for signature in (
            strategy_signature(_diversity(node).get("strategy_labels")) for node in nodes
        )
        if signature is not None
    ]
    lambdas = [
        float(_diversity(node).get("novelty_lambda_in_effect"))
        for node in nodes
        if isinstance(_diversity(node).get("novelty_lambda_in_effect"), (int, float))
    ]
    backends = sorted(
        {
            str(_diversity(node).get("embedding_backend"))
            for node in nodes
            if _diversity(node).get("embedding_backend")
        }
    )
    return {
        "run_id": run_id,
        "node_count": len(nodes),
        "mean_nn_distance": mean(distances),
        "strategy_label_entropy_bits": shannon_entropy(signatures),
        "novelty_lambda": lambdas[0] if lambdas else None,
        "embedding_backends": backends,
        "best_validation_score": _best_validation_score(nodes),
    }


def correlation_table(run_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Correlate novelty_lambda against diversity metrics and best score."""
    table: dict[str, Any] = {}
    for target in ("mean_nn_distance", "strategy_label_entropy_bits", "best_validation_score"):
        pairs = [
            (summary["novelty_lambda"], summary[target])
            for summary in run_summaries
            if summary["novelty_lambda"] is not None and summary[target] is not None
        ]
        table[f"novelty_lambda_vs_{target}"] = {
            "n_runs": len(pairs),
            "pearson_r": pearson([p[0] for p in pairs], [p[1] for p in pairs]),
        }
    return table


def build_report(paths: list[Path]) -> dict[str, Any]:
    nodes = iter_nodes(paths)
    by_run: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        by_run.setdefault(str(node.get("run_id", "unknown")), []).append(node)
    run_summaries = [summarize_run(run_id, run_nodes) for run_id, run_nodes in sorted(by_run.items())]
    return {
        "node_count": len(nodes),
        "run_count": len(run_summaries),
        "runs": run_summaries,
        "correlations": correlation_table(run_summaries),
        "notes": (
            "best_validation_score honors per-run metric_direction (min when minimize); "
            "correlations are computed on raw scores, so interpret sign with the metric "
            "direction in mind."
        ),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_report(report: dict[str, Any]) -> None:
    header = (
        "run_id",
        "nodes",
        "novelty_lambda",
        "mean_nn_dist",
        "label_entropy",
        "best_val_score",
        "backends",
    )
    rows = [
        (
            summary["run_id"],
            str(summary["node_count"]),
            _fmt(summary["novelty_lambda"]),
            _fmt(summary["mean_nn_distance"]),
            _fmt(summary["strategy_label_entropy_bits"]),
            _fmt(summary["best_validation_score"]),
            ",".join(summary["embedding_backends"]) or "-",
        )
        for summary in report["runs"]
    ]
    widths = [max(len(header[i]), *(len(row[i]) for row in rows)) if rows else len(header[i]) for i in range(len(header))]
    print("  ".join(header[i].ljust(widths[i]) for i in range(len(header))))
    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(header))))
    print()
    print("correlations (novelty_lambda vs ...):")
    for key, entry in report["correlations"].items():
        target = key.replace("novelty_lambda_vs_", "")
        print(f"  {target:32s} n={entry['n_runs']:<3d} pearson_r={_fmt(entry['pearson_r'])}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nodes_jsonl", nargs="+", type=Path, help="nodes.jsonl export path(s)")
    parser.add_argument("--json-out", type=Path, default=None, help="optional JSON report path")
    args = parser.parse_args(argv)

    missing = [path for path in args.nodes_jsonl if not path.exists()]
    if missing:
        print(f"error: missing input file(s): {', '.join(str(p) for p in missing)}", file=sys.stderr)
        return 1

    report = build_report(args.nodes_jsonl)
    print_report(report)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
