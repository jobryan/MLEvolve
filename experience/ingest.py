"""Offline ingestion of completed runs into the experience store.

Usage:
    python -m experience.ingest --store-dir /path/to/store RUN_DIR [RUN_DIR ...] \
        [--task-id ID] [--run-id RID] [--domain D] \
        [--metric-name M] [--metric-direction maximize|minimize] [--dry-run]

Each RUN_DIR is a completed MLEvolve run directory. The run's per-task global
memory records (workspace/global_memory/records.json) are converted into
ExperienceRecords with provenance and appended to the store. Duplicate
(task, run, node) records are skipped, so re-ingesting is safe.

--task-id may only be given with a single RUN_DIR; with several, the task id is
derived from each run's saved config (data_dir basename) to prevent mislabeling.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .record import ExperienceRecord
from .store import RECORDS_FILENAME, append_records, compute_snapshot_hash, load_records


def find_run_records_file(run_dir: Path) -> Optional[Path]:
    for candidate in (
        run_dir / "workspace" / "global_memory" / "records.json",
        run_dir / "global_memory" / "records.json",
    ):
        if candidate.exists():
            return candidate
    return None


def read_run_config(run_dir: Path) -> Dict[str, str]:
    """Best-effort read of exp_name/data_dir from the run's saved config."""
    for candidate in (run_dir / "logs" / "config.yaml", run_dir / "config.yaml"):
        if not candidate.exists():
            continue
        text = candidate.read_text(encoding="utf-8", errors="replace")
        try:
            import yaml

            cfg = yaml.safe_load(text) or {}
            return {"exp_name": str(cfg.get("exp_name") or ""), "data_dir": str(cfg.get("data_dir") or "")}
        except Exception:
            out = {}
            for key in ("exp_name", "data_dir"):
                m = re.search(rf"^{key}:\s*(.+)$", text, re.MULTILINE)
                out[key] = m.group(1).strip().strip("'\"") if m else ""
            return out
    return {"exp_name": "", "data_dir": ""}


def derive_task_id(run_config: Dict[str, str]) -> str:
    data_dir = run_config.get("data_dir", "")
    if not data_dir:
        return ""
    name = Path(data_dir).name
    # mle-bench prepared layouts end in .../<competition>/prepared/public
    if name in ("public", "prepared", "input", "data"):
        for part in reversed(Path(data_dir).parts):
            if part not in ("public", "prepared", "input", "data"):
                return part
    return name


def convert_run_records(
    run_records_file: Path,
    task_id: str,
    run_id: str,
    domain: str,
    metric_name: str,
    metric_direction: str,
) -> List[ExperienceRecord]:
    with open(run_records_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    converted = []
    for item in raw:
        source_id = item.get("record_id", "")
        title = item.get("title", "")
        stage = title.split(" - ")[0] if " - " in title else "unknown"
        converted.append(
            ExperienceRecord(
                record_id=f"{task_id}/{run_id}/{source_id}",
                task_id=task_id,
                run_id=run_id,
                stage=stage,
                description=item.get("description", ""),
                method=item.get("method", ""),
                label=item.get("label", 0),
                domain=domain,
                metric_name=metric_name,
                metric_direction=metric_direction,
                parent_metric=item.get("parent_metric"),
                current_metric=item.get("current_metric"),
                exec_time=item.get("exec_time"),
                parent_error=item.get("parent_error", ""),
                timestamp=item.get("timestamp"),
            )
        )
    return converted


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Completed run directories to ingest")
    parser.add_argument("--store-dir", required=True, type=Path, help="Persistent experience store directory")
    parser.add_argument("--task-id", default="", help="Competition id (single run dir only)")
    parser.add_argument("--run-id", default="", help="Run id override (single run dir only)")
    parser.add_argument("--domain", default="", help="Task domain tag, e.g. 'Tabular'")
    parser.add_argument("--metric-name", default="")
    parser.add_argument("--metric-direction", default="", choices=["", "maximize", "minimize"])
    parser.add_argument("--dry-run", action="store_true", help="Report what would be ingested without writing")
    args = parser.parse_args(argv)

    if len(args.run_dirs) > 1 and (args.task_id or args.run_id):
        parser.error("--task-id/--run-id may only be used with a single RUN_DIR")

    records_file = args.store_dir / RECORDS_FILENAME
    existing_ids = {r.record_id for r in load_records(records_file)}

    total_added, total_skipped = 0, 0
    for run_dir in args.run_dirs:
        run_records_file = find_run_records_file(run_dir)
        if run_records_file is None:
            print(f"[skip] {run_dir}: no global_memory/records.json found", file=sys.stderr)
            continue

        run_config = read_run_config(run_dir)
        task_id = (args.task_id or derive_task_id(run_config)).strip().lower()
        if not task_id:
            print(f"[skip] {run_dir}: task id underivable; pass --task-id", file=sys.stderr)
            continue
        run_id = args.run_id or run_config.get("exp_name") or run_dir.name

        converted = convert_run_records(
            run_records_file, task_id, run_id, args.domain, args.metric_name, args.metric_direction,
        )
        fresh = [r for r in converted if r.record_id not in existing_ids]
        skipped = len(converted) - len(fresh)

        if not args.dry_run and fresh:
            append_records(records_file, fresh)
        existing_ids.update(r.record_id for r in fresh)
        total_added += len(fresh)
        total_skipped += skipped
        print(f"[ok] {run_dir}: task={task_id} run={run_id} added={len(fresh)} skipped_dup={skipped}")

    action = "would add" if args.dry_run else "added"
    print(f"Done: {action} {total_added} records ({total_skipped} duplicates skipped)")
    print(f"Store snapshot: {compute_snapshot_hash(records_file)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
