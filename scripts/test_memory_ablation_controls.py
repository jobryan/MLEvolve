#!/usr/bin/env python3
"""Dependency-light checks for MLEvolve memory ablation controls."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.memory.ablation_controls import (
    get_child_memory,
    retrieve_global_memory,
)
from utils.ablation_export import AblationExporter


class FakeRecord:
    def __init__(self, record_id: str, label: int, title: str):
        self.record_id = record_id
        self.label = label
        self.title = title
        self.description = f"plan {record_id}"
        self.method = f"method {record_id}"


class FakeGlobalMemory:
    def __init__(self):
        self.records = [
            FakeRecord("success", 1, "improve - success"),
            FakeRecord("failure", -1, "improve - failure"),
            FakeRecord("debug_success", 1, "debug - success"),
        ]
        self.calls = []

    def retrieve_similar_records(self, **kwargs):
        self.calls.append(kwargs)
        label_filter = kwargs.get("label_filter")
        stage_filter = kwargs.get("stage_filter")
        rows = self.records
        if label_filter is not None:
            rows = [record for record in rows if record.label == label_filter]
        if stage_filter is not None:
            rows = [record for record in rows if record.title.startswith(f"{stage_filter} - ")]
        return [(record, 0.9 - index * 0.1) for index, record in enumerate(rows[: kwargs.get("top_k", 2)])]


class FakeNode:
    id = "parent"

    def __init__(self):
        self.fetch_called = False
        self.parent = None
        self.children = []

    def fetch_child_memory(self, include_code=False):
        self.fetch_called = True
        return "Attempt #1: prior experiment"


def make_agent(**ablation_overrides):
    ablation = {
        "enabled": True,
        "schema_version": "1.0",
        "run_id": "memory-test-run",
        "variant_id": "memory-test",
        "benchmark_track": "shared_mle_bench_lite",
        "task_id": "spooky-author-identification",
        "phase": "smoke",
        "task_manifest_version": "mle-bench-lite-v1",
        "variant_registry_version": "ablation-variants-v1",
        "child_memory": True,
        "global_memory_filter": "all",
        "dissimilar_guidance": False,
    }
    ablation.update(ablation_overrides)
    cfg = SimpleNamespace(
        ablation=SimpleNamespace(**ablation),
        agent=SimpleNamespace(
            seed=1,
            code=SimpleNamespace(model="synthetic-code-model"),
            feedback=SimpleNamespace(model="synthetic-feedback-model"),
        ),
        workspace_dir=tempfile.mkdtemp(prefix="memory-export-test-"),
        exp_name="synthetic",
        exp_id="synthetic-exp",
        eval="multi-class-log-loss",
    )
    return SimpleNamespace(
        cfg=cfg,
        acfg=SimpleNamespace(use_global_memory=True),
        global_memory=FakeGlobalMemory(),
        use_coldstart=False,
    )


def test_child_memory_disabled() -> None:
    agent = make_agent(child_memory=False)
    node = FakeNode()

    memory_text = get_child_memory(agent, node)

    assert memory_text == ""
    assert not node.fetch_called
    assert agent.current_memory_events[-1]["enabled"] is False
    assert agent.current_memory_events[-1]["event_type"] == "child_memory"


def test_global_memory_filters() -> None:
    success_agent = make_agent(global_memory_filter="success_only")
    success_records = retrieve_global_memory(
        success_agent,
        query_text="plan",
        label_filter=1,
        event_type="success_lookup",
    )
    excluded_failure_records = retrieve_global_memory(
        success_agent,
        query_text="plan",
        label_filter=-1,
        event_type="failure_lookup",
    )
    assert [record.record_id for record, _ in success_records] == ["success", "debug_success"]
    assert excluded_failure_records == []

    failure_agent = make_agent(global_memory_filter="failure_only")
    failure_records = retrieve_global_memory(
        failure_agent,
        query_text="plan",
        label_filter=None,
        event_type="failure_mode_lookup",
    )
    assert [record.record_id for record, _ in failure_records] == ["failure"]

    none_agent = make_agent(global_memory_filter="none")
    none_records = retrieve_global_memory(
        none_agent,
        query_text="plan",
        event_type="none_lookup",
    )
    assert none_records == []
    assert none_agent.global_memory.calls == []


def test_memory_events_export() -> None:
    agent = make_agent(global_memory_filter="success_only")
    node = SimpleNamespace(
        id="node-memory-001",
        parent=None,
        stage="improve",
        branch_id="branch-001",
        created_time="2026-06-23T00:00:00Z",
        finish_time="2026-06-23T00:00:12Z",
        prompt_input="Prompt with memory controls.",
        code="print('memory')\n",
        metric=SimpleNamespace(value=0.5, maximize=False),
        is_valid=True,
        is_buggy=False,
        children=[],
        exec_time=12.0,
        memory_events=[
            {
                "event_type": "success_lookup",
                "source": "global_memory",
                "enabled": True,
                "record_count": 1,
                "records": [{"record_id": "success", "label": 1, "stage": "improve", "score": 0.9}],
                "text_chars": 0,
                "reason": "global_memory_filter=success_only",
            }
        ],
    )

    exporter = AblationExporter(agent)
    exporter.export_node(node)
    node_path = Path(agent.cfg.workspace_dir) / "ablation_export/nodes.jsonl"
    record = json.loads(node_path.read_text().splitlines()[0])
    assert record["memory_sources"]["raw"]["event_count"] == 1
    embedding_path = ROOT / record["diversity"]["embedding_path"]
    assert embedding_path.exists()

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/validate_ablation_schema.py",
            "--schema",
            "configs/ablations/schema/node.schema.json",
            "--jsonl",
            str(node_path),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed.stdout.strip())
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> int:
    test_child_memory_disabled()
    test_global_memory_filters()
    test_memory_events_export()
    print("memory ablation control tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
