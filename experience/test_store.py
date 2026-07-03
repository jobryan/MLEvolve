"""Tests for the cross-run experience store (BM25-only path, no model downloads).

Run: python -m experience.test_store   (or pytest experience/test_store.py)
"""

import json
import tempfile
from pathlib import Path

from .ingest import main as ingest_main
from .record import ExperienceRecord
from .store import RECORDS_FILENAME, ExperienceStore, append_records, compute_snapshot_hash


def _make_record(task_id: str, run_id: str, node: str, label: int, description: str) -> ExperienceRecord:
    return ExperienceRecord(
        record_id=f"{task_id}/{run_id}/node_{node}",
        task_id=task_id,
        run_id=run_id,
        stage="improve",
        description=description,
        method=f"method for {description}",
        label=label,
        domain="Tabular",
        metric_name="auc-roc",
        metric_direction="maximize",
    )


def _seed_store(store_dir: Path) -> None:
    append_records(store_dir / RECORDS_FILENAME, [
        _make_record("task-a", "run1", "1", 1, "gradient boosting with target encoding on tabular features"),
        _make_record("task-a", "run1", "2", -1, "deep neural network overfit on small tabular dataset"),
        _make_record("task-b", "run2", "3", 1, "gradient boosting ensemble with cross validation for tabular data"),
        _make_record("task-c", "run3", "4", -1, "image augmentation pipeline with heavy rotations"),
    ])


def test_leakage_guard_and_retrieval():
    with tempfile.TemporaryDirectory() as tmp:
        store_dir = Path(tmp) / "store"
        _seed_store(store_dir)

        store = ExperienceStore(store_dir=str(store_dir), current_task_id="task-a")
        assert store.excluded_same_task == 2
        assert len(store.records) == 2
        assert all(r.task_id != "task-a" for r in store.records)

        results = store.retrieve("gradient boosting tabular", top_k=5)
        assert results, "BM25 should match the task-b record"
        assert results[0][0].task_id == "task-b"

        guarded = ExperienceStore(
            store_dir=str(store_dir), current_task_id="task-a", excluded_task_ids=["task-b"],
        )
        assert guarded.excluded_listed == 1
        assert [r.task_id for r in guarded.records] == ["task-c"]


def test_guidance_and_injection_log():
    with tempfile.TemporaryDirectory() as tmp:
        store_dir = Path(tmp) / "store"
        log_path = Path(tmp) / "logs" / "experience_injections.jsonl"
        _seed_store(store_dir)

        store = ExperienceStore(
            store_dir=str(store_dir), current_task_id="task-a", injection_log_path=log_path,
        )
        augmented = store.augment_memory_text(
            "existing within-run memory", "gradient boosting tabular ensemble", context_label="draft",
        )
        assert augmented.startswith("existing within-run memory")
        assert "Cross-Run Experience" in augmented
        assert "task-b" in augmented or "Tabular" in augmented

        entries = [json.loads(l) for l in log_path.read_text().splitlines()]
        assert len(entries) == 1
        assert entries[0]["context"] == "draft"
        assert entries[0]["snapshot_hash"] == store.snapshot_hash()
        assert entries[0]["excluded_same_task"] == 2
        assert all(r["task_id"] != "task-a" for r in entries[0]["retrieved"])

        # empty retrieval leaves memory text unchanged (but still logs)
        unchanged = store.augment_memory_text("memory", "zzz qqq nomatch", context_label="improve")
        assert unchanged == "memory"


def test_snapshot_hash_changes_on_append():
    with tempfile.TemporaryDirectory() as tmp:
        store_dir = Path(tmp) / "store"
        records_file = store_dir / RECORDS_FILENAME
        _seed_store(store_dir)
        h1 = compute_snapshot_hash(records_file)
        assert h1 == compute_snapshot_hash(records_file), "hash must be deterministic"
        append_records(records_file, [_make_record("task-d", "run4", "5", 0, "another approach")])
        assert compute_snapshot_hash(records_file) != h1


def test_ingest_cli_dedup_and_task_derivation():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "20260702_run"
        (run_dir / "workspace" / "global_memory").mkdir(parents=True)
        (run_dir / "logs").mkdir(parents=True)
        (run_dir / "workspace" / "global_memory" / "records.json").write_text(json.dumps([
            {"record_id": "node_abc", "title": "improve - abc", "description": "plan text",
             "method": "code summary", "label": 1, "current_metric": 0.9, "exec_time": 12.5},
            {"record_id": "node_def", "title": "debug - def", "description": "fix plan",
             "method": "fix summary", "label": 1, "parent_error": "KeyError: 'target'"},
        ]))
        (run_dir / "logs" / "config.yaml").write_text(
            "exp_name: 20260702_run\ndata_dir: /data/spooky-author-identification/prepared/public\n"
        )

        store_dir = Path(tmp) / "store"
        rc = ingest_main([str(run_dir), "--store-dir", str(store_dir), "--domain", "Text Classification"])
        assert rc == 0

        store = ExperienceStore(store_dir=str(store_dir), current_task_id="other-task")
        assert len(store.records) == 2
        rec = next(r for r in store.records if r.stage == "improve")
        assert rec.task_id == "spooky-author-identification"
        assert rec.run_id == "20260702_run"
        assert rec.current_metric == 0.9
        debug_rec = next(r for r in store.records if r.stage == "debug")
        assert "KeyError" in debug_rec.search_text()

        # re-ingesting is a no-op
        ingest_main([str(run_dir), "--store-dir", str(store_dir)])
        assert len(ExperienceStore(store_dir=str(store_dir), current_task_id="other-task").records) == 2

        # leakage guard applies to ingested records
        same_task = ExperienceStore(store_dir=str(store_dir), current_task_id="spooky-author-identification")
        assert len(same_task.records) == 0
        assert same_task.excluded_same_task == 2


if __name__ == "__main__":
    for fn in (
        test_leakage_guard_and_retrieval,
        test_guidance_and_injection_log,
        test_snapshot_hash_changes_on_append,
        test_ingest_cli_dedup_and_task_derivation,
    ):
        fn()
        print(f"PASS {fn.__name__}")
    print("All experience store tests passed.")
