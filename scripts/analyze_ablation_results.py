#!/usr/bin/env python3
"""Analyze exported ablation tables and emit statistics/report artifacts."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


JsonDict = dict[str, Any]
VariantKey = tuple[str, str, str]


def read_csv_rows(path: Path) -> list[JsonDict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[JsonDict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def to_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "pass", "success"}:
        return True
    if text in {"false", "0", "no", "fail", "invalid_submission"}:
        return False
    return None


def mean(values: Iterable[Any]) -> float | None:
    numeric = [value for value in (to_float(item) for item in values) if value is not None]
    if not numeric:
        return None
    return statistics.fmean(numeric)


def sum_values(values: Iterable[Any]) -> float:
    return sum(value for value in (to_float(item) for item in values) if value is not None)


def key_from(row: JsonDict) -> VariantKey:
    return (str(row.get("system", "")), str(row.get("variant_id", "")), str(row.get("phase", "")))


def key_label(key: VariantKey) -> str:
    return f"{key[0]}::{key[1]}::{key[2]}"


def comparison_score(row: JsonDict) -> float | None:
    valid = to_bool(row.get("valid_submission"))
    score = to_float(row.get("normalized_score"))
    if valid is False:
        return 0.0
    return score


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def shannon_entropy(labels: list[str]) -> float | None:
    if not labels:
        return None
    counts = Counter(labels)
    total = sum(counts.values())
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def variant_metrics(runs: list[JsonDict], nodes: list[JsonDict]) -> list[JsonDict]:
    grouped_runs: dict[VariantKey, list[JsonDict]] = defaultdict(list)
    grouped_nodes: dict[VariantKey, list[JsonDict]] = defaultdict(list)
    for row in runs:
        grouped_runs[key_from(row)].append(row)
    for row in nodes:
        grouped_nodes[key_from(row)].append(row)

    baseline_by_system_phase: dict[tuple[str, str], float | None] = {}
    baseline_ids = {
        "mlevolve": "default_mcgs",
        "ai_scientist_v2": "stage_as_operator_default",
    }

    rows: list[JsonDict] = []
    for key, group in sorted(grouped_runs.items()):
        valid_values = [to_bool(row.get("valid_submission")) for row in group]
        valid_count = sum(1 for value in valid_values if value is True)
        medal_values = [to_bool(row.get("medal_achieved")) for row in group if row.get("medal_achieved") != ""]
        effective_scores = [score for score in (comparison_score(row) for row in group) if score is not None]
        normalized_scores = [to_float(row.get("normalized_score")) for row in group]
        normalized_scores = [score for score in normalized_scores if score is not None]
        node_group = grouped_nodes.get(key, [])
        success_nodes = [node for node in node_group if node.get("status") == "success"]
        total_cost = sum_values(row.get("actual_cost_usd") for row in group)
        mean_effective = statistics.fmean(effective_scores) if effective_scores else None

        row = {
            "system": key[0],
            "variant_id": key[1],
            "phase": key[2],
            "run_count": len(group),
            "task_count": len({row.get("task_id") for row in group}),
            "seed_count": len({row.get("seed") for row in group}),
            "valid_submission_count": valid_count,
            "valid_submission_rate": valid_count / len(group) if group else None,
            "medal_count": sum(1 for value in medal_values if value is True) if medal_values else "",
            "medal_rate": (sum(1 for value in medal_values if value is True) / len(medal_values)) if medal_values else "",
            "mean_normalized_score": statistics.fmean(normalized_scores) if normalized_scores else None,
            "mean_effective_normalized_score": mean_effective,
            "best_effective_normalized_score": max(effective_scores) if effective_scores else None,
            "total_cost_usd": total_cost,
            "total_wall_time_seconds": sum_values(row.get("actual_wall_time_seconds") for row in group),
            "node_count": len(node_group),
            "valid_node_count": len(success_nodes),
            "cost_per_valid_node": (total_cost / len(success_nodes)) if success_nodes else None,
            "cost_per_score_improvement": None,
        }
        rows.append(row)

        if baseline_ids.get(key[0]) == key[1]:
            baseline_by_system_phase[(key[0], key[2])] = mean_effective

    for row in rows:
        baseline = baseline_by_system_phase.get((row["system"], row["phase"]))
        current = to_float(row.get("mean_effective_normalized_score"))
        total_cost = to_float(row.get("total_cost_usd"))
        if baseline is not None and current is not None and total_cost is not None:
            delta = current - baseline
            row["score_delta_vs_system_baseline"] = delta
            row["cost_per_score_improvement"] = total_cost / delta if delta > 0 else None
        else:
            row["score_delta_vs_system_baseline"] = None

    return rows


def pairwise_win_rates(runs: list[JsonDict]) -> list[JsonDict]:
    by_task_seed: dict[tuple[str, str, str], dict[VariantKey, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in runs:
        score = comparison_score(row)
        if score is None:
            continue
        task_seed = (str(row.get("phase")), str(row.get("task_id")), str(row.get("seed")))
        by_task_seed[task_seed][key_from(row)].append(score)

    pair_counts: dict[tuple[VariantKey, VariantKey], Counter[str]] = defaultdict(Counter)
    for variants in by_task_seed.values():
        averaged = {key: statistics.fmean(scores) for key, scores in variants.items()}
        keys = sorted(averaged)
        for i, left in enumerate(keys):
            for right in keys[i + 1 :]:
                pair = (left, right)
                if averaged[left] > averaged[right]:
                    pair_counts[pair]["left_wins"] += 1
                elif averaged[left] < averaged[right]:
                    pair_counts[pair]["right_wins"] += 1
                else:
                    pair_counts[pair]["ties"] += 1
                pair_counts[pair]["paired_count"] += 1

    rows: list[JsonDict] = []
    for (left, right), counts in sorted(pair_counts.items(), key=lambda item: (key_label(item[0][0]), key_label(item[0][1]))):
        paired = counts["paired_count"]
        rows.append(
            {
                "variant_a": key_label(left),
                "variant_b": key_label(right),
                "paired_count": paired,
                "a_wins": counts["left_wins"],
                "b_wins": counts["right_wins"],
                "ties": counts["ties"],
                "a_win_rate": counts["left_wins"] / paired if paired else None,
            }
        )
    return rows


def bootstrap_intervals(runs: list[JsonDict], iterations: int, seed: int) -> list[JsonDict]:
    rng = random.Random(seed)
    by_variant_task: dict[VariantKey, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in runs:
        score = comparison_score(row)
        if score is None:
            continue
        by_variant_task[key_from(row)][str(row.get("task_id"))].append(score)

    rows: list[JsonDict] = []
    for key, task_scores in sorted(by_variant_task.items()):
        task_means = [statistics.fmean(values) for values in task_scores.values() if values]
        if not task_means:
            continue
        samples: list[float] = []
        for _ in range(iterations):
            resampled = [rng.choice(task_means) for _ in task_means]
            samples.append(statistics.fmean(resampled))
        rows.append(
            {
                "system": key[0],
                "variant_id": key[1],
                "phase": key[2],
                "task_count": len(task_means),
                "mean_effective_normalized_score": statistics.fmean(task_means),
                "ci_low_95": percentile(samples, 0.025),
                "ci_high_95": percentile(samples, 0.975),
                "bootstrap_iterations": iterations,
            }
        )
    return rows


def validation_gap(nodes: list[JsonDict]) -> list[JsonDict]:
    grouped: dict[VariantKey, list[float]] = defaultdict(list)
    for node in nodes:
        validation = to_float(node.get("validation_score"))
        final = to_float(node.get("final_score"))
        if validation is None or final is None:
            continue
        direction = str(node.get("metric_direction") or "")
        if direction == "minimize":
            gap = final - validation
        elif direction == "maximize":
            gap = validation - final
        else:
            gap = abs(final - validation)
        grouped[key_from(node)].append(gap)

    rows: list[JsonDict] = []
    for key, gaps in sorted(grouped.items()):
        rows.append(
            {
                "system": key[0],
                "variant_id": key[1],
                "phase": key[2],
                "node_count": len(gaps),
                "mean_overfit_gap": statistics.fmean(gaps),
                "max_overfit_gap": max(gaps),
            }
        )
    return rows


def diversity_summary(nodes: list[JsonDict]) -> list[JsonDict]:
    grouped: dict[VariantKey, list[JsonDict]] = defaultdict(list)
    for node in nodes:
        grouped[key_from(node)].append(node)

    rows: list[JsonDict] = []
    for key, group in sorted(grouped.items()):
        labels: list[str] = []
        novelty_values: list[float] = []
        for node in group:
            labels.extend(label for label in str(node.get("strategy_labels") or "").split(";") if label)
            novelty = to_float(node.get("novelty_distance"))
            if novelty is not None:
                novelty_values.append(novelty)
        rows.append(
            {
                "system": key[0],
                "variant_id": key[1],
                "phase": key[2],
                "node_count": len(group),
                "mean_nearest_neighbor_distance": statistics.fmean(novelty_values) if novelty_values else None,
                "max_nearest_neighbor_distance": max(novelty_values) if novelty_values else None,
                "strategy_label_entropy": shannon_entropy(labels),
                "unique_strategy_label_count": len(set(labels)),
            }
        )
    return rows


def recommendation_rows(metrics: list[JsonDict]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for row in metrics:
        valid_rate = to_float(row.get("valid_submission_rate")) or 0.0
        seed_count = int(row.get("seed_count") or 0)
        effective = to_float(row.get("mean_effective_normalized_score"))
        delta = to_float(row.get("score_delta_vs_system_baseline"))
        if valid_rate == 0:
            decision = "repair_or_drop"
            reason = "No valid submissions in available runs."
        elif seed_count < 2:
            decision = "needs_more_evidence"
            reason = "Only one seed is available; do not make a stable keep/remove claim."
        elif delta is not None and delta < -0.2:
            decision = "modify_or_drop"
            reason = "Effective normalized score is materially below the system baseline."
        else:
            decision = "keep_provisionally"
            reason = "Valid submissions exist and current evidence does not show a large regression."

        rows.append(
            {
                "system": row["system"],
                "variant_id": row["variant_id"],
                "phase": row["phase"],
                "draft_decision": decision,
                "evidence": reason,
                "valid_submission_rate": valid_rate,
                "mean_effective_normalized_score": effective,
            }
        )
    return rows


def svg_bar(path: Path, rows: list[JsonDict], label_field: str, value_field: str, title: str) -> None:
    values = [(str(row[label_field]), to_float(row.get(value_field))) for row in rows]
    values = [(label, value) for label, value in values if value is not None]
    width, height = 900, 420
    margin = 70
    max_value = max([value for _, value in values] or [1.0])
    bar_width = max(16, int((width - margin * 2) / max(len(values), 1) * 0.7))
    step = (width - margin * 2) / max(len(values), 1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin}" y="32" font-family="Arial" font-size="20" fill="#111">{html.escape(title)}</text>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#333"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#333"/>',
    ]
    for index, (label, value) in enumerate(values):
        bar_height = 0 if max_value == 0 else (value / max_value) * (height - margin * 2)
        x = margin + index * step + (step - bar_width) / 2
        y = height - margin - bar_height
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="#2f6f9f"/>')
        parts.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{height - margin + 18}" '
            f'font-family="Arial" font-size="10" text-anchor="middle" fill="#111">'
            f'{html.escape(label[:18])}</text>'
        )
        parts.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{y - 6:.1f}" '
            f'font-family="Arial" font-size="10" text-anchor="middle" fill="#111">{value:.3f}</text>'
        )
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def svg_scatter(path: Path, rows: list[JsonDict], x_field: str, y_field: str, label_field: str, title: str) -> None:
    points = [
        (str(row[label_field]), to_float(row.get(x_field)), to_float(row.get(y_field)))
        for row in rows
    ]
    points = [(label, x, y) for label, x, y in points if x is not None and y is not None]
    width, height = 900, 480
    margin = 70
    xs = [x for _, x, _ in points] or [0.0, 1.0]
    ys = [y for _, _, y in points] or [0.0, 1.0]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if min_x == max_x:
        max_x = min_x + 1.0
    if min_y == max_y:
        max_y = min_y + 1.0

    def scale_x(value: float) -> float:
        return margin + (value - min_x) / (max_x - min_x) * (width - margin * 2)

    def scale_y(value: float) -> float:
        return height - margin - (value - min_y) / (max_y - min_y) * (height - margin * 2)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin}" y="32" font-family="Arial" font-size="20" fill="#111">{html.escape(title)}</text>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#333"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#333"/>',
        f'<text x="{width / 2}" y="{height - 18}" font-family="Arial" font-size="12" text-anchor="middle">{html.escape(x_field)}</text>',
        f'<text x="18" y="{height / 2}" font-family="Arial" font-size="12" transform="rotate(-90 18 {height / 2})" text-anchor="middle">{html.escape(y_field)}</text>',
    ]
    for label, x, y in points:
        sx, sy = scale_x(x), scale_y(y)
        parts.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="5" fill="#9f5f2f"/>')
        parts.append(
            f'<text x="{sx + 8:.1f}" y="{sy - 8:.1f}" font-family="Arial" font-size="10" fill="#111">'
            f'{html.escape(label[:26])}</text>'
        )
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_report(path: Path, metrics: list[JsonDict], recs: list[JsonDict], pairwise_rows: list[JsonDict]) -> None:
    ranked = sorted(
        metrics,
        key=lambda row: to_float(row.get("mean_effective_normalized_score")) or -1.0,
        reverse=True,
    )
    lines = [
        "# Ablation Analysis Summary",
        "",
        "Status: generated from exported shared-schema tables.",
        "",
        "## Scope",
        "",
        "This analysis computes valid-submission rate, optional medal rate, normalized/effective score, paired win rate, paired bootstrap confidence intervals, validation-test gap, operator outcomes, diversity entropy, novelty distance, and cost metrics.",
        "",
        "Mixed-effects modeling is not included in this dependency-light script; paired bootstrap confidence intervals over tasks are used instead.",
        "",
        "## Top Variants By Effective Normalized Score",
        "",
        "| Rank | System | Variant | Phase | Valid Rate | Effective Score | Cost |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for index, row in enumerate(ranked[:10], 1):
        lines.append(
            f"| {index} | {row['system']} | {row['variant_id']} | {row['phase']} | "
            f"{to_float(row.get('valid_submission_rate')) or 0:.3f} | "
            f"{to_float(row.get('mean_effective_normalized_score')) or 0:.3f} | "
            f"{to_float(row.get('total_cost_usd')) or 0:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Draft Recommendations",
            "",
            "| System | Variant | Phase | Decision | Evidence |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in recs:
        lines.append(
            f"| {row['system']} | {row['variant_id']} | {row['phase']} | "
            f"{row['draft_decision']} | {row['evidence']} |"
        )

    lines.extend(
        [
            "",
            "## Pairwise Coverage",
            "",
            f"Pairwise comparisons with at least one shared task/seed: {len(pairwise_rows)}.",
            "",
            "Interpretation note: invalid submissions are assigned effective normalized score 0.0 for pairwise comparisons.",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory created by export_ablation_results.py.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-iterations", default=1000, type=int)
    parser.add_argument("--bootstrap-seed", default=7, type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs = read_csv_rows(args.input_dir / "runs.csv")
    nodes = read_csv_rows(args.input_dir / "nodes.csv")
    operator_rows = read_csv_rows(args.input_dir / "operator_stats.csv")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = variant_metrics(runs, nodes)
    pairwise = pairwise_win_rates(runs)
    bootstrap = bootstrap_intervals(runs, args.bootstrap_iterations, args.bootstrap_seed)
    gaps = validation_gap(nodes)
    diversity = diversity_summary(nodes)
    recs = recommendation_rows(metrics)

    write_csv(
        args.output_dir / "variant_metrics.csv",
        metrics,
        [
            "system",
            "variant_id",
            "phase",
            "run_count",
            "task_count",
            "seed_count",
            "valid_submission_count",
            "valid_submission_rate",
            "medal_count",
            "medal_rate",
            "mean_normalized_score",
            "mean_effective_normalized_score",
            "best_effective_normalized_score",
            "score_delta_vs_system_baseline",
            "total_cost_usd",
            "total_wall_time_seconds",
            "node_count",
            "valid_node_count",
            "cost_per_valid_node",
            "cost_per_score_improvement",
        ],
    )
    write_csv(
        args.output_dir / "pairwise_win_rates.csv",
        pairwise,
        ["variant_a", "variant_b", "paired_count", "a_wins", "b_wins", "ties", "a_win_rate"],
    )
    write_csv(
        args.output_dir / "bootstrap_intervals.csv",
        bootstrap,
        [
            "system",
            "variant_id",
            "phase",
            "task_count",
            "mean_effective_normalized_score",
            "ci_low_95",
            "ci_high_95",
            "bootstrap_iterations",
        ],
    )
    write_csv(
        args.output_dir / "validation_gap.csv",
        gaps,
        ["system", "variant_id", "phase", "node_count", "mean_overfit_gap", "max_overfit_gap"],
    )
    write_csv(
        args.output_dir / "diversity_summary.csv",
        diversity,
        [
            "system",
            "variant_id",
            "phase",
            "node_count",
            "mean_nearest_neighbor_distance",
            "max_nearest_neighbor_distance",
            "strategy_label_entropy",
            "unique_strategy_label_count",
        ],
    )
    write_csv(
        args.output_dir / "draft_recommendations.csv",
        recs,
        [
            "system",
            "variant_id",
            "phase",
            "draft_decision",
            "evidence",
            "valid_submission_rate",
            "mean_effective_normalized_score",
        ],
    )

    plot_rows = [
        {**row, "label": f"{row['system']}:{row['variant_id']}"}
        for row in metrics
    ]
    diversity_by_key = {key_from(row): row for row in diversity}
    diversity_plot_rows = []
    for row in metrics:
        key = key_from(row)
        div = diversity_by_key.get(key, {})
        diversity_plot_rows.append(
            {
                "label": f"{row['system']}:{row['variant_id']}",
                "mean_nearest_neighbor_distance": div.get("mean_nearest_neighbor_distance"),
                "mean_effective_normalized_score": row.get("mean_effective_normalized_score"),
            }
        )

    operator_plot_rows = []
    for row in operator_rows:
        node_count = to_float(row.get("node_count")) or 0.0
        success = to_float(row.get("success_count")) or 0.0
        operator_plot_rows.append(
            {
                "label": f"{row.get('operator')}:{row.get('variant_id')}",
                "success_rate": success / node_count if node_count else None,
            }
        )

    svg_bar(
        args.output_dir / "plots/variant_effective_score.svg",
        plot_rows,
        "label",
        "mean_effective_normalized_score",
        "Variant Effective Normalized Score",
    )
    svg_scatter(
        args.output_dir / "plots/cost_quality_frontier.svg",
        plot_rows,
        "total_cost_usd",
        "mean_effective_normalized_score",
        "label",
        "Cost Quality Frontier",
    )
    svg_bar(
        args.output_dir / "plots/operator_success.svg",
        operator_plot_rows,
        "label",
        "success_rate",
        "Operator Success Rate",
    )
    svg_scatter(
        args.output_dir / "plots/diversity_vs_performance.svg",
        diversity_plot_rows,
        "mean_nearest_neighbor_distance",
        "mean_effective_normalized_score",
        "label",
        "Diversity vs Performance",
    )

    summary = {
        "run_count": len(runs),
        "node_count": len(nodes),
        "variant_count": len(metrics),
        "pairwise_comparison_count": len(pairwise),
        "bootstrap_iterations": args.bootstrap_iterations,
        "limitations": [
            "Uses paired bootstrap over tasks instead of mixed-effects modeling.",
            "Invalid submissions receive effective normalized score 0.0 for pairwise/effective-score summaries.",
            "Medal rate is blank unless medal_achieved is present in exported run metrics.",
        ],
    }
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(args.output_dir / "ablation_analysis.md", metrics, recs, pairwise)

    print(f"analysis written to {args.output_dir}")
    print(f"variants={len(metrics)} pairwise={len(pairwise)} nodes={len(nodes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
