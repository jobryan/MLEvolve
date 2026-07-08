"""Optional ablation run/node export helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from utils.diversity_novelty import summarize_node_diversity

if TYPE_CHECKING:
    from engine.search_node import SearchNode


OPERATOR_BY_STAGE = {
    "root": "Draft",
    "draft": "Draft",
    "debug": "Debug",
    "improve": "Improve",
    "evolution": "Evolution",
    "fusion": "Fusion/Crossover",
    "fusion_draft": "Fusion/Crossover",
}


class AblationExporter:
    """Writes schema-compatible MLEvolve ablation records when enabled."""

    def __init__(self, agent: Any):
        self.agent = agent
        self.cfg = agent.cfg
        self.acfg = getattr(self.cfg, "ablation", None)
        self.enabled = bool(getattr(self.acfg, "enabled", False))
        self.export_dir = Path(self.cfg.workspace_dir) / "ablation_export"
        self.prompts_dir = self.export_dir / "prompts"
        self.code_dir = self.export_dir / "code"
        self.embeddings_dir = self.export_dir / "embeddings"
        self.submissions_dir = Path(self.cfg.workspace_dir) / "submission"
        self.runs_path = self.export_dir / "runs.jsonl"
        self.nodes_path = self.export_dir / "nodes.jsonl"

        if self.enabled:
            self.prompts_dir.mkdir(parents=True, exist_ok=True)
            self.code_dir.mkdir(parents=True, exist_ok=True)
            self.embeddings_dir.mkdir(parents=True, exist_ok=True)
            self.write_run_record(status="running")

    @property
    def run_id(self) -> str:
        configured = getattr(self.acfg, "run_id", "") if self.acfg is not None else ""
        return configured or str(getattr(self.cfg, "exp_name", "mlevolve-run"))

    def _relative(self, path: Path | None) -> str | None:
        if path is None:
            return None
        try:
            return str(path.resolve().relative_to(Path.cwd().resolve()))
        except ValueError:
            return str(path)

    def _append_jsonl(self, path: Path, record: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def write_run_record(self, status: str) -> None:
        if not self.enabled:
            return
        record = {
            "schema_version": getattr(self.acfg, "schema_version", "1.0"),
            "run_id": self.run_id,
            "system": "mlevolve",
            "benchmark_track": getattr(self.acfg, "benchmark_track", "diagnostic"),
            "task_id": getattr(self.acfg, "task_id", "") or str(getattr(self.cfg, "exp_id", "")),
            "task_manifest_version": getattr(self.acfg, "task_manifest_version", ""),
            "variant_id": getattr(self.acfg, "variant_id", "manual"),
            "phase": getattr(self.acfg, "phase", "smoke"),
            "seed": self._seed(),
            "status": status,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "completed_at": None,
            "git_commit": None,
            "git_branch": None,
            "command": None,
            "output_dir": self._relative(Path(self.cfg.workspace_dir)),
            "budget": {},
            "actual": {},
            "model_routing": {
                "code": self._stage_model("code"),
                "feedback": self._stage_model("feedback"),
            },
            "config_overrides": {},
            "resource_policy": {},
            "grader": {},
            "audit_summary": {},
            "metrics": {},
            "artifacts": {
                "runs_path": self._relative(self.runs_path),
                "nodes_path": self._relative(self.nodes_path),
            },
            "error": None,
            "notes": "MLEvolve ablation export run record.",
        }
        self._append_jsonl(self.runs_path, record)

    def save_prompt_and_code(self, node: "SearchNode") -> None:
        if not self.enabled:
            return
        if node.prompt_input:
            prompt_path = self.prompts_dir / f"{node.id}.md"
            prompt_path.write_text(node.prompt_input, encoding="utf-8")
        if node.code:
            code_path = self.code_dir / f"{node.id}.py"
            code_path.write_text(node.code, encoding="utf-8")

    def export_node(self, node: "SearchNode", status: str | None = None) -> None:
        if not self.enabled:
            return
        self.save_prompt_and_code(node)

        prompt_path = self.prompts_dir / f"{node.id}.md"
        code_path = self.code_dir / f"{node.id}.py"
        submission_path = self.submissions_dir / f"submission_{node.id}.csv"
        metric = getattr(node, "metric", None)
        parent = getattr(node, "parent", None)
        node_status = status or self._node_status(node)
        stage = getattr(node, "stage", "improve")
        diversity = self._diversity(node)

        record = {
            "schema_version": getattr(self.acfg, "schema_version", "1.0"),
            "node_id": node.id,
            "run_id": self.run_id,
            "system": "mlevolve",
            "task_id": getattr(self.acfg, "task_id", "") or str(getattr(self.cfg, "exp_id", "")),
            "variant_id": getattr(self.acfg, "variant_id", "manual"),
            "seed": self._seed(),
            "operator": OPERATOR_BY_STAGE.get(stage, "Improve"),
            "stage": stage,
            "parent_node_ids": [parent.id] if parent is not None else [],
            "reference_node_ids": [],
            "branch_id": str(node.branch_id) if node.branch_id is not None else None,
            "depth": self._depth(node),
            "status": node_status,
            "created_at": getattr(node, "created_time", None)
            or time.strftime(
                "%Y-%m-%dT%H:%M:%S",
                time.localtime(getattr(node, "ctime", time.time())),
            ),
            "completed_at": getattr(node, "finish_time", None),
            "prompt_paths": [self._relative(prompt_path)] if prompt_path.exists() else [],
            "generated_code_path": self._relative(code_path) if code_path.exists() else None,
            "submission_path": self._relative(submission_path) if submission_path.exists() else None,
            "grader_output_path": None,
            "audit_output_path": None,
            "metric_name": str(getattr(self.cfg, "eval", "") or "") or None,
            "metric_direction": self._metric_direction(metric),
            "validation_score": getattr(metric, "value", None),
            "final_score": None,
            "normalized_score": None,
            "wall_time_seconds": getattr(node, "exec_time", None),
            "cost": {},
            "token_usage": {},
            "memory_sources": self._memory_sources(node),
            "selection": getattr(node, "selection_rationale", None)
            or getattr(parent, "_selection_rationale", None)
            or {},
            "diversity": diversity,
            "artifacts": {
                "code_summary": getattr(node, "code_summary", None),
                "work_dir": getattr(node, "work_dir", None),
            },
            "error": self._error(node),
            "notes": None,
        }
        self._append_jsonl(self.nodes_path, record)

    def _agent_cfg(self) -> Any:
        return getattr(self.cfg, "agent", None)

    def _seed(self) -> int:
        return int(getattr(self._agent_cfg(), "seed", 0))

    def _stage_model(self, stage_name: str) -> str | None:
        stage_cfg = getattr(self._agent_cfg(), stage_name, None)
        return getattr(stage_cfg, "model", None)

    def _node_status(self, node: "SearchNode") -> str:
        if getattr(node, "is_valid", None) is False:
            return "invalid_submission"
        if getattr(node, "is_buggy", None) is True:
            return "runtime_error" if getattr(node, "exc_type", None) else "invalid_submission"
        if getattr(node, "is_buggy", None) is False:
            return "success"
        return "running"

    def _metric_direction(self, metric: Any) -> str | None:
        maximize = getattr(metric, "maximize", None)
        if maximize is None:
            return None
        return "maximize" if maximize else "minimize"

    def _depth(self, node: "SearchNode") -> int:
        depth = 0
        current = node.parent
        while current is not None:
            depth += 1
            current = current.parent
        return depth

    def _memory_sources(self, node: "SearchNode") -> dict[str, Any]:
        events = list(getattr(node, "memory_events", []) or [])
        return {
            "child_history_available": bool(node.parent and node.parent.children),
            "global_retrieval_enabled": bool(getattr(self.agent, "global_memory", None)),
            "journal_summary": None,
            "coldstart_enabled": bool(getattr(self.agent, "use_coldstart", False)),
            "raw": {
                "events": events,
                "event_count": len(events),
            },
        }

    def _previous_nodes(self) -> list[Any]:
        journal = getattr(self.agent, "journal", None)
        nodes = getattr(journal, "nodes", None)
        if nodes is None:
            return []
        return list(nodes)

    def _novelty_lambda(self) -> float:
        try:
            return float(getattr(self.acfg, "novelty_lambda", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _diversity(self, node: "SearchNode") -> dict[str, Any]:
        summary = summarize_node_diversity(node, self._previous_nodes())
        embedding = summary.pop("embedding")
        embedding_backend = summary.get("embedding_backend", "hashing-token-v1")
        embedding_path = self.embeddings_dir / f"{node.id}.json"
        embedding_path.write_text(
            json.dumps(
                {
                    "node_id": node.id,
                    "embedding_model": embedding_backend,
                    "embedding": embedding,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        summary["embedding_path"] = self._relative(embedding_path)
        summary["novelty_lambda_in_effect"] = self._novelty_lambda()
        return summary

    def _error(self, node: "SearchNode") -> dict[str, Any] | None:
        if not getattr(node, "is_buggy", False) and getattr(node, "is_valid", True) is not False:
            return None
        return {
            "type": getattr(node, "exc_type", None) or "invalid_or_buggy",
            "message": str(getattr(node, "exc_info", None) or getattr(node, "analysis", "") or ""),
        }


def maybe_get_exporter(agent: Any) -> AblationExporter | None:
    exporter = getattr(agent, "ablation_exporter", None)
    if exporter is not None and getattr(exporter, "enabled", False):
        return exporter
    return None
