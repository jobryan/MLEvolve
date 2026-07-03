"""Tests for the cross-run experience store (BM25-only path, no model downloads).

Run: python -m experience.test_store   (or pytest experience/test_store.py)
"""

import json
import tempfile
from pathlib import Path

from .entries import BugBookEntry, Lesson, SolutionEntry
from .ingest import main as ingest_main
from .mining import error_signature, index_solutions, mine_bugbook
from .record import ExperienceRecord
from .store import (
    BUGBOOK_FILENAME,
    LESSONS_FILENAME,
    RECORDS_FILENAME,
    SOLUTIONS_FILENAME,
    ExperienceStore,
    append_entries,
    append_records,
    compute_snapshot_hash,
)


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


def _write_fixture_journal(path: Path) -> None:
    """Journal with one buggy draft fixed by a debug child, plus an improve node."""
    nodes = [
        {"id": "aaa", "stage": "draft", "is_buggy": True, "plan": "first draft",
         "_term_out": ["Traceback (most recent call last):\n",
                       "  File \"/tmp/work/runfile.py\", line 42, in <module>\n",
                       "KeyError: 'target_column_7'\n"],
         "exc_type": "KeyError", "code": "x=1"},
        {"id": "bbb", "stage": "debug", "is_buggy": False,
         "plan": "Use the sample submission columns instead of guessing target names.",
         "code_summary": "read sample_submission.csv header for target columns",
         "_term_out": ["ok\n"], "code": "x=2", "metric": {"value": 0.5}},
        {"id": "ccc", "stage": "improve", "is_buggy": False, "plan": "add cv folds",
         "_term_out": ["ok\n"], "code": "x=3", "metric": {"value": 0.6}},
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"nodes": nodes, "node2parent": {"bbb": "aaa", "ccc": "bbb"}}))


def test_error_signature_normalization():
    sig1 = error_signature("Traceback ...\nKeyError: 'target_column_7'\n")
    sig2 = error_signature("Traceback ...\nKeyError: 'target_column_9'\n")
    assert sig1 == sig2 == "KeyError: '<val>'"
    sig3 = error_signature("ValueError: shape mismatch 128 vs 256 in /opt/data/train.csv")
    assert "<n>" in sig3 and "<path>" in sig3 and sig3.startswith("ValueError")
    assert error_signature("no error here") == ""


def test_bugbook_mining_and_guidance():
    with tempfile.TemporaryDirectory() as tmp:
        journal = Path(tmp) / "run" / "logs" / "journal.json"
        _write_fixture_journal(journal)
        bugs = mine_bugbook(journal, "task-x", "run9", domain="Tabular")
        assert len(bugs) == 1
        assert bugs[0].error_signature == "KeyError: '<val>'"
        assert "sample submission" in bugs[0].fix_plan

        store_dir = Path(tmp) / "store"
        append_entries(store_dir / BUGBOOK_FILENAME, bugs)
        store = ExperienceStore(store_dir=str(store_dir), current_task_id="other-task")
        guidance = store.get_bugbook_guidance("KeyError: 'some_other_col'")
        assert "KeyError" in guidance and "sample submission" in guidance
        # leakage guard applies to the bugbook too
        same = ExperienceStore(store_dir=str(store_dir), current_task_id="task-x")
        assert same.get_bugbook_guidance("KeyError: 'some_other_col'") == ""
        # mechanism flag off -> no guidance
        off = ExperienceStore(store_dir=str(store_dir), current_task_id="other-task", use_bugbook=False)
        assert off.get_bugbook_guidance("KeyError: 'x'") == ""


def test_lessons_in_guidance_and_flags():
    with tempfile.TemporaryDirectory() as tmp:
        store_dir = Path(tmp) / "store"
        _seed_store(store_dir)
        append_entries(store_dir / LESSONS_FILENAME, [
            Lesson(lesson_id="task-b/run2/lesson_0", task_id="task-b", run_id="run2",
                   text="On small tabular datasets, gradient boosting beats neural networks.",
                   scope_tags=["tabular", "small-dataset"], confidence="high"),
        ])
        store = ExperienceStore(store_dir=str(store_dir), current_task_id="task-a")
        guidance = store.generate_guidance_prompt("small tabular dataset gradient boosting")
        assert "Distilled lessons" in guidance and "gradient boosting beats" in guidance
        no_lessons = ExperienceStore(store_dir=str(store_dir), current_task_id="task-a", use_lessons=False)
        assert "Distilled lessons" not in no_lessons.generate_guidance_prompt("small tabular gradient boosting")
        no_episodic = ExperienceStore(store_dir=str(store_dir), current_task_id="task-a", use_episodic=False)
        g = no_episodic.generate_guidance_prompt("small tabular dataset gradient boosting")
        assert "Distilled lessons" in g and "Approaches that worked" not in g


def test_solutions_index_and_guidance():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        top1 = run_dir / "workspace" / "top_solution" / "top1"
        top1.mkdir(parents=True)
        (top1 / "solution.py").write_text("import xgboost\n# tabular pipeline\n")
        (top1 / "metric.txt").write_text("Metric: 0.91\nMaximize: True\n")

        store_dir = Path(tmp) / "store"
        entries = index_solutions(run_dir, store_dir, "task-y", "run3", domain="Tabular", metric_name="auc-roc")
        assert len(entries) == 1 and entries[0].metric_value == 0.91 and entries[0].metric_maximize is True
        append_entries(store_dir / SOLUTIONS_FILENAME, entries)

        store = ExperienceStore(store_dir=str(store_dir), current_task_id="other-task")
        guidance = store.get_solution_guidance("Tabular auc-roc competition")
        assert "xgboost" in guidance and "adapt" in guidance.lower()
        same = ExperienceStore(store_dir=str(store_dir), current_task_id="task-y")
        assert same.get_solution_guidance("Tabular auc-roc competition") == ""


def test_snapshot_hash_covers_all_store_files():
    with tempfile.TemporaryDirectory() as tmp:
        store_dir = Path(tmp) / "store"
        _seed_store(store_dir)
        h1 = compute_snapshot_hash(store_dir)
        append_entries(store_dir / LESSONS_FILENAME, [
            Lesson(lesson_id="t/r/lesson_0", task_id="t", run_id="r", text="x", scope_tags=[]),
        ])
        h2 = compute_snapshot_hash(store_dir)
        assert h1 != h2, "lessons.jsonl must affect the snapshot hash"


def test_ingest_with_bugbook_and_solutions():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "20260702_run"
        (run_dir / "workspace" / "global_memory").mkdir(parents=True)
        (run_dir / "workspace" / "global_memory" / "records.json").write_text(json.dumps([
            {"record_id": "node_abc", "title": "improve - abc", "description": "plan", "method": "m", "label": 1},
        ]))
        (run_dir / "logs").mkdir(parents=True)
        (run_dir / "logs" / "config.yaml").write_text("exp_name: r1\ndata_dir: /d/task-z/prepared/public\n")
        _write_fixture_journal(run_dir / "logs" / "journal.json")
        top1 = run_dir / "workspace" / "top_solution" / "top1"
        top1.mkdir(parents=True)
        (top1 / "solution.py").write_text("print('solution')\n")
        (top1 / "metric.txt").write_text("Metric: 0.7\nMaximize: True\n")

        store_dir = Path(tmp) / "store"
        rc = ingest_main([str(run_dir), "--store-dir", str(store_dir), "--with-bugbook", "--with-solutions"])
        assert rc == 0
        store = ExperienceStore(store_dir=str(store_dir), current_task_id="other")
        assert len(store.records) == 1 and len(store.bugbook) == 1 and len(store.solutions) == 1
        # idempotent
        ingest_main([str(run_dir), "--store-dir", str(store_dir), "--with-bugbook", "--with-solutions"])
        store2 = ExperienceStore(store_dir=str(store_dir), current_task_id="other")
        assert len(store2.records) == 1 and len(store2.bugbook) == 1 and len(store2.solutions) == 1


if __name__ == "__main__":
    for fn in (
        test_leakage_guard_and_retrieval,
        test_guidance_and_injection_log,
        test_snapshot_hash_changes_on_append,
        test_ingest_cli_dedup_and_task_derivation,
        test_error_signature_normalization,
        test_bugbook_mining_and_guidance,
        test_lessons_in_guidance_and_flags,
        test_solutions_index_and_guidance,
        test_snapshot_hash_covers_all_store_files,
        test_ingest_with_bugbook_and_solutions,
    ):
        fn()
        print(f"PASS {fn.__name__}")
    print("All experience store tests passed.")
