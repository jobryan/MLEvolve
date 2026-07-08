#!/usr/bin/env python3
"""Export shared ablation JSONL logs into analysis-ready CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validate_ablation_schema import iter_jsonl, load_json, validate_record  # noqa: E402


DEFAULT_TASK_MANIFEST = ROOT / "configs/ablations/tasks/mle_bench_lite.json"
RUN_SCHEMA = ROOT / "configs/ablations/schema/run.schema.json"
NODE_SCHEMA = ROOT / "configs/ablations/schema/node.schema.json"

RUN_FIELDS = [
    "run_id",
    "system",
    "benchmark_track",
    "task_id",
    "task_name",
    "task_domain",
    "metric_name",
    "metric_direction",
    "variant_id",
    "phase",
    "seed",
    "status",
    "valid_submission",
    "audit_status",
    "audit_failure_reasons",
    "best_validation_score",
    "final_score",
    "normalized_score",
    "medal_achieved",
    "baseline_score",
    "score_delta",
    "actual_nodes",
    "actual_wall_time_seconds",
    "actual_cost_usd",
    "budget_wall_time_seconds",
    "budget_max_nodes",
    "budget_max_cost_usd",
    "started_at",
    "completed_at",
    "output_dir",
    "error_type",
    "error_message",
]

NODE_FIELDS = [
    "node_id",
    "run_id",
    "system",
    "task_id",
    "task_name",
    "metric_name",
    "metric_direction",
    "variant_id",
    "phase",
    "seed",
    "operator",
    "stage",
    "status",
    "parent_count",
    "reference_count",
    "depth",
    "validation_score",
    "final_score",
    "normalized_score",
    "wall_time_seconds",
    "cost_usd",
    "input_tokens",
    "output_tokens",
    "selection_policy",
    "selection_score",
    "novelty_distance",
    "strategy_labels",
    "memory_child_history_count",
    "memory_global_retrieval_count",
    "has_journal_summary",
    "created_at",
    "completed_at",
    "submission_path",
    "error_type",
    "error_message",
]

OPERATOR_FIELDS = [
    "system",
    "variant_id",
    "phase",
    "operator",
    "node_count",
    "success_count",
    "invalid_submission_count",
    "error_count",
    "mean_normalized_score",
    "best_normalized_score",
    "mean_wall_time_seconds",
    "total_cost_usd",
    "mean_input_tokens",
    "mean_output_tokens",
]

VARIANT_FIELDS = [
    "system",
    "variant_id",
    "phase",
    "run_count",
    "task_count",
    "seed_count",
    "success_count",
    "valid_submission_count",
    "valid_submission_rate",
    "audit_pass_count",
    "audit_fail_count",
    "audit_pass_rate",
    "mean_normalized_score",
    "best_normalized_score",
    "mean_final_score",
    "total_nodes",
    "total_cost_usd",
    "total_wall_time_seconds",
    "status_counts",
    "tasks",
]

AUDIT_FIELDS = [
    "system",
    "variant_id",
    "phase",
    "audit_status",
    "failure_reason",
    "count",
]


JsonDict = dict[str, Any]


def nested_get(record: JsonDict, *keys: str, default: Any = None) -> Any:
    current: Any = record
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def compact_json(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def mean(values: Iterable[Any]) -> float | None:
    numeric = [value for value in (as_float(item) for item in values) if value is not None]
    if not numeric:
        return None
    return statistics.fmean(numeric)


def sum_values(values: Iterable[Any]) -> float:
    return sum(value for value in (as_float(item) for item in values) if value is not None)


def best_normalized(values: Iterable[Any]) -> float | None:
    numeric = [value for value in (as_float(item) for item in values) if value is not None]
    if not numeric:
        return None
    return max(numeric)


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes", "pass", "success"}
    return bool(value)


def load_records(paths: list[Path], schema_path: Path, label: str) -> tuple[list[JsonDict], list[str]]:
    schema = load_json(schema_path)
    records: list[JsonDict] = []
    errors: list[str] = []

    for path in paths:
        if not path.exists():
            errors.append(f"{label}: missing file {path}")
            continue
        try:
            parsed = iter_jsonl(path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        for line_no, record in parsed:
            source = f"{path}:{line_no}"
            errors.extend(validate_record(schema, record, source))
            records.append(record)

    return records, errors


def load_tasks(path: Path) -> tuple[dict[str, JsonDict], list[str]]:
    if not path.exists():
        return {}, [f"task manifest missing: {path}"]

    payload = load_json(path)
    tasks: dict[str, JsonDict] = {}
    errors: list[str] = []
    for task in payload.get("tasks", []):
        task_id = task.get("id")
        if not task_id:
            errors.append(f"task manifest contains task without id: {task!r}")
            continue
        if task_id in tasks:
            errors.append(f"task manifest contains duplicate task id {task_id!r}")
        tasks[task_id] = task
    return tasks, errors


def audit_status(record: JsonDict) -> str:
    value = nested_get(record, "audit_summary", "status")
    if value is None:
        return ""
    return str(value)


def audit_failures(record: JsonDict) -> list[str]:
    summary = record.get("audit_summary") or {}
    failures = summary.get("failure_reasons")
    if failures is None:
        failures = summary.get("blocking_failures")
    if failures is None:
        return []
    if isinstance(failures, list):
        return [str(item) for item in failures]
    return [str(failures)]


def valid_submission(record: JsonDict) -> bool:
    metric_value = nested_get(record, "metrics", "valid_submission")
    if metric_value is not None:
        return boolish(metric_value)
    if audit_status(record) == "pass":
        return True
    return record.get("status") == "success" and nested_get(record, "metrics", "final_score") is not None


def error_type(record: JsonDict) -> str:
    error = record.get("error")
    if isinstance(error, dict):
        return str(error.get("type") or "")
    return ""


def error_message(record: JsonDict) -> str:
    error = record.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or "")
    return ""


def data_quality(
    runs: list[JsonDict],
    nodes: list[JsonDict],
    tasks: dict[str, JsonDict],
    schema_errors: list[str],
) -> JsonDict:
    errors = list(schema_errors)
    warnings: list[str] = []

    run_ids = [str(run.get("run_id")) for run in runs]
    node_ids = [str(node.get("node_id")) for node in nodes]
    for run_id, count in Counter(run_ids).items():
        if count > 1:
            errors.append(f"duplicate run_id {run_id!r}")
    for node_id, count in Counter(node_ids).items():
        if count > 1:
            errors.append(f"duplicate node_id {node_id!r}")

    run_by_id = {str(run.get("run_id")): run for run in runs}
    node_by_id = {str(node.get("node_id")): node for node in nodes}
    nodes_by_run: dict[str, list[JsonDict]] = defaultdict(list)

    for run in runs:
        task_id = str(run.get("task_id"))
        if task_id not in tasks:
            errors.append(f"run {run.get('run_id')!r} references unknown task_id {task_id!r}")

    for node in nodes:
        run_id = str(node.get("run_id"))
        nodes_by_run[run_id].append(node)
        run = run_by_id.get(run_id)
        if run is None:
            errors.append(f"node {node.get('node_id')!r} references missing run_id {run_id!r}")
        else:
            for field in ("system", "task_id", "variant_id", "seed"):
                if node.get(field) != run.get(field):
                    errors.append(
                        f"node {node.get('node_id')!r} field {field!r}={node.get(field)!r} "
                        f"does not match run {run_id!r} value {run.get(field)!r}"
                    )

        task_id = str(node.get("task_id"))
        if task_id not in tasks:
            errors.append(f"node {node.get('node_id')!r} references unknown task_id {task_id!r}")

        for parent_id in node.get("parent_node_ids", []):
            if parent_id not in node_by_id:
                errors.append(f"node {node.get('node_id')!r} references missing parent node {parent_id!r}")
                continue
            parent = node_by_id[parent_id]
            if parent.get("run_id") != run_id:
                errors.append(
                    f"node {node.get('node_id')!r} parent {parent_id!r} belongs to run {parent.get('run_id')!r}"
                )

        for reference_id in node.get("reference_node_ids", []):
            if reference_id not in node_by_id:
                warnings.append(f"node {node.get('node_id')!r} references missing reference node {reference_id!r}")

    for run_id, run in run_by_id.items():
        actual_nodes = as_int(nested_get(run, "actual", "nodes"))
        if actual_nodes is not None and actual_nodes != len(nodes_by_run.get(run_id, [])):
            warnings.append(
                f"run {run_id!r} reports actual.nodes={actual_nodes} but has "
                f"{len(nodes_by_run.get(run_id, []))} node records"
            )
        if run.get("status") == "success" and audit_status(run) == "fail":
            warnings.append(f"run {run_id!r} succeeded but audit_summary.status is fail")

    return {
        "status": "fail" if errors else "pass",
        "run_count": len(runs),
        "node_count": len(nodes),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
    }


def run_rows(runs: list[JsonDict], tasks: dict[str, JsonDict]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for run in runs:
        task = tasks.get(str(run.get("task_id")), {})
        row = {
            "run_id": run.get("run_id"),
            "system": run.get("system"),
            "benchmark_track": run.get("benchmark_track"),
            "task_id": run.get("task_id"),
            "task_name": task.get("name", ""),
            "task_domain": task.get("domain", ""),
            "metric_name": task.get("metric_name") or nested_get(run, "metrics", "metric_name", default=""),
            "metric_direction": task.get("metric_direction") or nested_get(run, "metrics", "metric_direction", default=""),
            "variant_id": run.get("variant_id"),
            "phase": run.get("phase"),
            "seed": run.get("seed"),
            "status": run.get("status"),
            "valid_submission": valid_submission(run),
            "audit_status": audit_status(run),
            "audit_failure_reasons": ";".join(audit_failures(run)),
            "best_validation_score": nested_get(run, "metrics", "best_validation_score"),
            "final_score": nested_get(run, "metrics", "final_score"),
            "normalized_score": nested_get(run, "metrics", "normalized_score"),
            "medal_achieved": nested_get(run, "metrics", "medal_achieved"),
            "baseline_score": nested_get(run, "metrics", "baseline_score"),
            "score_delta": nested_get(run, "metrics", "score_delta"),
            "actual_nodes": nested_get(run, "actual", "nodes"),
            "actual_wall_time_seconds": nested_get(run, "actual", "wall_time_seconds"),
            "actual_cost_usd": nested_get(run, "actual", "cost_usd"),
            "budget_wall_time_seconds": nested_get(run, "budget", "wall_time_seconds"),
            "budget_max_nodes": nested_get(run, "budget", "max_nodes"),
            "budget_max_cost_usd": nested_get(run, "budget", "max_cost_usd"),
            "started_at": run.get("started_at"),
            "completed_at": run.get("completed_at"),
            "output_dir": run.get("output_dir"),
            "error_type": error_type(run),
            "error_message": error_message(run),
        }
        rows.append(row)
    return rows


def node_rows(nodes: list[JsonDict], tasks: dict[str, JsonDict], run_by_id: dict[str, JsonDict]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for node in nodes:
        task = tasks.get(str(node.get("task_id")), {})
        memory = node.get("memory_sources") or {}
        selection = node.get("selection") or {}
        diversity = node.get("diversity") or {}
        token_usage = node.get("token_usage") or {}
        cost = node.get("cost") or {}
        run = run_by_id.get(str(node.get("run_id")), {})
        row = {
            "node_id": node.get("node_id"),
            "run_id": node.get("run_id"),
            "system": node.get("system"),
            "task_id": node.get("task_id"),
            "task_name": task.get("name", ""),
            "metric_name": node.get("metric_name") or task.get("metric_name", ""),
            "metric_direction": node.get("metric_direction") or task.get("metric_direction", ""),
            "variant_id": node.get("variant_id"),
            "phase": run.get("phase", ""),
            "seed": node.get("seed"),
            "operator": node.get("operator"),
            "stage": node.get("stage"),
            "status": node.get("status"),
            "parent_count": len(node.get("parent_node_ids") or []),
            "reference_count": len(node.get("reference_node_ids") or []),
            "depth": node.get("depth"),
            "validation_score": node.get("validation_score"),
            "final_score": node.get("final_score"),
            "normalized_score": node.get("normalized_score"),
            "wall_time_seconds": node.get("wall_time_seconds"),
            "cost_usd": cost.get("usd"),
            "input_tokens": token_usage.get("input_tokens"),
            "output_tokens": token_usage.get("output_tokens"),
            "selection_policy": selection.get("policy"),
            "selection_score": selection.get("selection_score"),
            "novelty_distance": diversity.get("nearest_neighbor_distance"),
            "strategy_labels": ";".join(str(label) for label in (diversity.get("strategy_labels") or [])),
            "memory_child_history_count": len(memory.get("child_history") or []),
            "memory_global_retrieval_count": len(memory.get("global_retrieval") or []),
            "has_journal_summary": bool(memory.get("journal_summary")),
            "created_at": node.get("created_at"),
            "completed_at": node.get("completed_at"),
            "submission_path": node.get("submission_path"),
            "error_type": error_type(node),
            "error_message": error_message(node),
        }
        rows.append(row)
    return rows


def operator_stats(nodes: list[JsonDict], run_by_id: dict[str, JsonDict]) -> list[JsonDict]:
    groups: dict[tuple[str, str, str, str], list[JsonDict]] = defaultdict(list)
    for node in nodes:
        run = run_by_id.get(str(node.get("run_id")), {})
        key = (
            str(node.get("system")),
            str(node.get("variant_id")),
            str(run.get("phase", "")),
            str(node.get("operator")),
        )
        groups[key].append(node)

    rows: list[JsonDict] = []
    for (system, variant_id, phase, operator), group in sorted(groups.items()):
        status_counts = Counter(str(node.get("status")) for node in group)
        rows.append(
            {
                "system": system,
                "variant_id": variant_id,
                "phase": phase,
                "operator": operator,
                "node_count": len(group),
                "success_count": status_counts.get("success", 0),
                "invalid_submission_count": status_counts.get("invalid_submission", 0),
                "error_count": sum(status_counts[status] for status in ("runtime_error", "grader_error", "audit_failed")),
                "mean_normalized_score": mean(node.get("normalized_score") for node in group),
                "best_normalized_score": best_normalized(node.get("normalized_score") for node in group),
                "mean_wall_time_seconds": mean(node.get("wall_time_seconds") for node in group),
                "total_cost_usd": sum_values(nested_get(node, "cost", "usd") for node in group),
                "mean_input_tokens": mean(nested_get(node, "token_usage", "input_tokens") for node in group),
                "mean_output_tokens": mean(nested_get(node, "token_usage", "output_tokens") for node in group),
            }
        )
    return rows


def variant_summary(runs: list[JsonDict], nodes_by_run: dict[str, list[JsonDict]]) -> list[JsonDict]:
    groups: dict[tuple[str, str, str], list[JsonDict]] = defaultdict(list)
    for run in runs:
        key = (str(run.get("system")), str(run.get("variant_id")), str(run.get("phase")))
        groups[key].append(run)

    rows: list[JsonDict] = []
    for (system, variant_id, phase), group in sorted(groups.items()):
        status_counts = Counter(str(run.get("status")) for run in group)
        valid_count = sum(1 for run in group if valid_submission(run))
        audit_pass_count = sum(1 for run in group if audit_status(run) == "pass")
        audit_fail_count = sum(1 for run in group if audit_status(run) == "fail")
        total_nodes = sum(len(nodes_by_run.get(str(run.get("run_id")), [])) for run in group)
        rows.append(
            {
                "system": system,
                "variant_id": variant_id,
                "phase": phase,
                "run_count": len(group),
                "task_count": len({run.get("task_id") for run in group}),
                "seed_count": len({run.get("seed") for run in group}),
                "success_count": status_counts.get("success", 0),
                "valid_submission_count": valid_count,
                "valid_submission_rate": valid_count / len(group) if group else None,
                "audit_pass_count": audit_pass_count,
                "audit_fail_count": audit_fail_count,
                "audit_pass_rate": audit_pass_count / (audit_pass_count + audit_fail_count)
                if (audit_pass_count + audit_fail_count)
                else None,
                "mean_normalized_score": mean(nested_get(run, "metrics", "normalized_score") for run in group),
                "best_normalized_score": best_normalized(nested_get(run, "metrics", "normalized_score") for run in group),
                "mean_final_score": mean(nested_get(run, "metrics", "final_score") for run in group),
                "total_nodes": total_nodes,
                "total_cost_usd": sum_values(nested_get(run, "actual", "cost_usd") for run in group),
                "total_wall_time_seconds": sum_values(nested_get(run, "actual", "wall_time_seconds") for run in group),
                "status_counts": compact_json(dict(sorted(status_counts.items()))),
                "tasks": ";".join(sorted({str(run.get("task_id")) for run in group})),
            }
        )
    return rows


def audit_summary(runs: list[JsonDict]) -> list[JsonDict]:
    counter: Counter[tuple[str, str, str, str, str]] = Counter()
    for run in runs:
        status = audit_status(run) or "missing"
        failures = audit_failures(run) or [""]
        for failure in failures:
            counter[
                (
                    str(run.get("system")),
                    str(run.get("variant_id")),
                    str(run.get("phase")),
                    status,
                    failure,
                )
            ] += 1

    rows = []
    for (system, variant_id, phase, status, failure), count in sorted(counter.items()):
        rows.append(
            {
                "system": system,
                "variant_id": variant_id,
                "phase": phase,
                "audit_status": status,
                "failure_reason": failure,
                "count": count,
            }
        )
    return rows


def write_csv(path: Path, rows: list[JsonDict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _normalize_run_id(run_id: str) -> str:
    return run_id.strip().replace("_", "-")


def load_grades(paths: list[Path]) -> dict[str, JsonDict]:
    """Grade rows from grade_ablation_submissions.py backfill JSONL, keyed by run id."""
    grades: dict[str, JsonDict] = {}
    for path in paths or []:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                run_id = str(row.get("run_id") or "")
                if run_id:
                    grades[_normalize_run_id(run_id)] = row
    return grades


def apply_grades(runs: list[JsonDict], grades: dict[str, JsonDict]) -> dict[str, int]:
    """Join MLE-bench grader results into run metrics (metrics.final_score)."""
    matched = 0
    for run in runs:
        row = grades.get(_normalize_run_id(str(run.get("run_id"))))
        if row is None:
            continue
        metrics = run.setdefault("metrics", {})
        if row.get("score") is not None:
            metrics["final_score"] = row["score"]
        for key in ("valid_submission", "any_medal", "above_median", "is_lower_better"):
            if row.get(key) is not None:
                metrics[f"grader_{key}"] = row[key]
        matched += 1
    return {"grade_rows": len(grades), "matched_runs": matched}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-jsonl", required=True, action="append", type=Path)
    parser.add_argument("--nodes-jsonl", required=True, action="append", type=Path)
    parser.add_argument(
        "--grades-jsonl",
        action="append",
        type=Path,
        default=[],
        help="Optional grades JSONL from grade_ablation_submissions.py; joined into metrics.final_score by run_id.",
    )
    parser.add_argument("--task-manifest", default=DEFAULT_TASK_MANIFEST, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks, task_errors = load_tasks(args.task_manifest)
    runs, run_errors = load_records(args.runs_jsonl, RUN_SCHEMA, "runs")
    nodes, node_errors = load_records(args.nodes_jsonl, NODE_SCHEMA, "nodes")
    grade_join = apply_grades(runs, load_grades(args.grades_jsonl))
    quality = data_quality(runs, nodes, tasks, task_errors + run_errors + node_errors)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "data_quality.json").write_text(
        json.dumps(quality, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if quality["status"] != "pass":
        print(f"data quality failed; see {args.output_dir / 'data_quality.json'}", file=sys.stderr)
        return 1

    run_by_id = {str(run.get("run_id")): run for run in runs}
    nodes_by_run: dict[str, list[JsonDict]] = defaultdict(list)
    for node in nodes:
        nodes_by_run[str(node.get("run_id"))].append(node)

    outputs = {
        "runs.csv": (run_rows(runs, tasks), RUN_FIELDS),
        "nodes.csv": (node_rows(nodes, tasks, run_by_id), NODE_FIELDS),
        "operator_stats.csv": (operator_stats(nodes, run_by_id), OPERATOR_FIELDS),
        "variant_summary.csv": (variant_summary(runs, nodes_by_run), VARIANT_FIELDS),
        "audit_summary.csv": (audit_summary(runs), AUDIT_FIELDS),
    }
    for filename, (rows, fields) in outputs.items():
        write_csv(args.output_dir / filename, rows, fields)

    print(f"exported ablation tables to {args.output_dir}")
    print(f"runs={len(runs)} nodes={len(nodes)} warnings={quality['warning_count']}")
    if args.grades_jsonl:
        print(
            f"grades: rows={grade_join['grade_rows']} matched_runs={grade_join['matched_runs']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
