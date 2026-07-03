"""Experience record: one node-level experience from a past run.

Extends the per-run MemRecord shape with provenance (task/run/node) and task
metadata so records can be retrieved across tasks with a leakage guard.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class ExperienceRecord:
    record_id: str            # globally unique: "{task_id}/{run_id}/{source_record_id}"
    task_id: str              # competition/task the record came from (leakage-guard key)
    run_id: str               # source run (exp_name or user-supplied)
    stage: str                # draft / improve / debug / evolution / fusion / ...
    description: str          # plan text
    method: str               # code summary
    label: int                # 1 success, 0 neutral, -1 failure
    domain: str = ""          # e.g. "Tabular", "Image Classification"
    metric_name: str = ""
    metric_direction: str = ""  # "maximize" / "minimize"
    parent_metric: Optional[float] = None
    current_metric: Optional[float] = None
    exec_time: Optional[float] = None
    parent_error: str = ""    # for debug records: the error that was fixed
    timestamp: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperienceRecord":
        """Build from dict (e.g. loaded from JSONL). Extra keys are ignored."""
        return cls(
            record_id=data.get("record_id", ""),
            task_id=data.get("task_id", ""),
            run_id=data.get("run_id", ""),
            stage=data.get("stage", "unknown"),
            description=data.get("description", ""),
            method=data.get("method", ""),
            label=data.get("label", 0),
            domain=data.get("domain", ""),
            metric_name=data.get("metric_name", ""),
            metric_direction=data.get("metric_direction", ""),
            parent_metric=data.get("parent_metric"),
            current_metric=data.get("current_metric"),
            exec_time=data.get("exec_time"),
            parent_error=data.get("parent_error", ""),
            timestamp=data.get("timestamp"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, "")}

    def search_text(self) -> str:
        """Text used for retrieval indexing (mirrors GlobalMemoryLayer)."""
        if self.stage == "debug" and self.parent_error:
            return self.parent_error
        return f"{self.description}\n{self.method}"
