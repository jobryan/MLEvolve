#!/usr/bin/env python3
"""Generate audit-focused promotion gates from exported ablation results."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]
VariantKey = tuple[str, str, str]


GATE_FIELDS = [
    "system",
    "variant_id",
    "phase",
    "run_count",
    "seed_count",
    "success_count",
    "valid_submission_count",
    "invalid_submission_count",
    "audit_pass_count",
    "audit_fail_count",
    "unaudited_count",
    "best_normalized_score",
    "promotion_decision",
    "reason",
    "audit_failure_reasons",
]


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
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "pass", "success"}:
        return True
    if text in {"false", "0", "no", "fail", "invalid_submission"}:
        return False
    return None


def key_from(row: JsonDict) -> VariantKey:
    return (str(row.get("system", "")), str(row.get("variant_id", "")), str(row.get("phase", "")))


def split_reasons(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    return [item for item in str(value).split(";") if item]


def gate_reason(
    run_count: int,
    seed_count: int,
    valid_count: int,
    invalid_count: int,
    audit_fail_count: int,
    unaudited_count: int,
    reasons: list[str],
    min_confirmation_seeds: int,
) -> tuple[str, str]:
    if run_count == 0:
        return "no_data", "No runs are available."
    if invalid_count == run_count or valid_count == 0:
        return "do_not_promote_invalid", "No valid submissions are available for this variant."
    if audit_fail_count > 0:
        reason_text = ", ".join(sorted(set(reasons))) or "unspecified audit failure"
        return "blocked_by_audit", f"At least one candidate failed audit: {reason_text}."
    if unaudited_count > 0:
        return "provisional_unaudited", "At least one run is missing audit status."
    if seed_count < min_confirmation_seeds:
        return "needs_confirmation", f"Only {seed_count} seed(s) are available; require {min_confirmation_seeds}."
    return "promotion_eligible", "Audit passed and confirmation seed requirement is met."


def build_gate_rows(runs: list[JsonDict], min_confirmation_seeds: int) -> list[JsonDict]:
    grouped: dict[VariantKey, list[JsonDict]] = defaultdict(list)
    for row in runs:
        grouped[key_from(row)].append(row)

    rows: list[JsonDict] = []
    for key, group in sorted(grouped.items()):
        valid_count = sum(1 for row in group if to_bool(row.get("valid_submission")) is True)
        invalid_count = sum(1 for row in group if row.get("status") == "invalid_submission")
        success_count = sum(1 for row in group if row.get("status") == "success")
        audit_pass_count = sum(1 for row in group if row.get("audit_status") == "pass")
        audit_fail_count = sum(1 for row in group if row.get("audit_status") == "fail")
        unaudited_count = sum(1 for row in group if row.get("audit_status") in {"", None, "missing"})
        reasons: list[str] = []
        scores: list[float] = []
        for row in group:
            reasons.extend(split_reasons(row.get("audit_failure_reasons")))
            score = to_float(row.get("normalized_score"))
            if score is not None:
                scores.append(score)

        decision, reason = gate_reason(
            run_count=len(group),
            seed_count=len({row.get("seed") for row in group}),
            valid_count=valid_count,
            invalid_count=invalid_count,
            audit_fail_count=audit_fail_count,
            unaudited_count=unaudited_count,
            reasons=reasons,
            min_confirmation_seeds=min_confirmation_seeds,
        )

        rows.append(
            {
                "system": key[0],
                "variant_id": key[1],
                "phase": key[2],
                "run_count": len(group),
                "seed_count": len({row.get("seed") for row in group}),
                "success_count": success_count,
                "valid_submission_count": valid_count,
                "invalid_submission_count": invalid_count,
                "audit_pass_count": audit_pass_count,
                "audit_fail_count": audit_fail_count,
                "unaudited_count": unaudited_count,
                "best_normalized_score": max(scores) if scores else "",
                "promotion_decision": decision,
                "reason": reason,
                "audit_failure_reasons": ";".join(sorted(set(reasons))),
            }
        )
    return rows


def write_report(path: Path, gate_rows: list[JsonDict], runs: list[JsonDict]) -> None:
    decision_counts = Counter(row["promotion_decision"] for row in gate_rows)
    invalid_runs = [row for row in runs if row.get("status") == "invalid_submission"]
    audit_failed_runs = [row for row in runs if row.get("audit_status") == "fail" and row.get("status") != "invalid_submission"]
    unaudited_runs = [row for row in runs if row.get("audit_status") in {"", None, "missing"}]

    reason_counts: Counter[str] = Counter()
    for row in runs:
        for reason in split_reasons(row.get("audit_failure_reasons")):
            reason_counts[reason] += 1

    lines = [
        "# Ablation Audit Report",
        "",
        "Status: generated from exported run-level audit fields.",
        "",
        "## Promotion Gate Summary",
        "",
        f"- Variants assessed: {len(gate_rows)}.",
        f"- Promotion eligible: {decision_counts.get('promotion_eligible', 0)}.",
        f"- Needs confirmation: {decision_counts.get('needs_confirmation', 0)}.",
        f"- Provisional because unaudited: {decision_counts.get('provisional_unaudited', 0)}.",
        f"- Blocked by audit: {decision_counts.get('blocked_by_audit', 0)}.",
        f"- Invalid/no valid submissions: {decision_counts.get('do_not_promote_invalid', 0)}.",
        "",
        "## Gate Decisions",
        "",
        "| System | Variant | Phase | Decision | Valid | Audit Fail | Unaudited | Reason |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in gate_rows:
        lines.append(
            f"| {row['system']} | {row['variant_id']} | {row['phase']} | {row['promotion_decision']} | "
            f"{row['valid_submission_count']} | {row['audit_fail_count']} | {row['unaudited_count']} | {row['reason']} |"
        )

    lines.extend(
        [
            "",
            "## Invalid Submissions vs Audit Failures",
            "",
            f"- Invalid submission runs: {len(invalid_runs)}.",
            f"- Audit-failed runs with non-invalid status: {len(audit_failed_runs)}.",
            f"- Unaudited runs: {len(unaudited_runs)}.",
            "",
            "Invalid submissions mean the system did not produce a valid benchmark artifact. Audit failures mean an artifact or run exists but failed a hardening check such as leakage, forbidden data access, format mismatch, or runtime-limit violation.",
            "",
            "## Audit Failure Reasons",
            "",
        ]
    )
    if reason_counts:
        for reason, count in sorted(reason_counts.items()):
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- None recorded.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory created by export_ablation_results.py.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-confirmation-seeds", default=2, type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs = read_csv_rows(args.input_dir / "runs.csv")
    gate_rows = build_gate_rows(runs, args.min_confirmation_seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "promotion_gate.csv", gate_rows, GATE_FIELDS)
    write_report(args.output_dir / "audit_report.md", gate_rows, runs)
    print(f"audit report written to {args.output_dir}")
    print(f"variants={len(gate_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
