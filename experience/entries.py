"""Semantic (M2) and procedural (M3) experience entries.

Lesson       — distilled claim from a run's search tree (LLM reflection).
BugBookEntry — error-signature -> fix mined deterministically from debug chains.
SolutionEntry — indexed top solution from a past run (pipeline skeleton).

All carry task_id provenance so the store's leakage guard applies uniformly.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Lesson:
    lesson_id: str            # "{task_id}/{run_id}/lesson_{n}"
    task_id: str
    run_id: str
    text: str                 # the claim, self-contained
    scope_tags: List[str]     # e.g. ["tabular", "small-dataset", "ensembling"]
    confidence: str = ""      # high / medium / low (LLM self-assessed)
    provenance_node_ids: str = ""  # comma-separated source node ids
    domain: str = ""
    timestamp: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Lesson":
        return cls(
            lesson_id=data.get("lesson_id", ""),
            task_id=data.get("task_id", ""),
            run_id=data.get("run_id", ""),
            text=data.get("text", ""),
            scope_tags=list(data.get("scope_tags") or []),
            confidence=data.get("confidence", ""),
            provenance_node_ids=data.get("provenance_node_ids", ""),
            domain=data.get("domain", ""),
            timestamp=data.get("timestamp"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, "", [])}

    def search_text(self) -> str:
        return f"{self.text}\n{' '.join(self.scope_tags)}\n{self.domain}"


@dataclass
class BugBookEntry:
    entry_id: str             # "{task_id}/{run_id}/bug_{n}"
    task_id: str
    run_id: str
    error_signature: str      # normalized "ErrType: message-shape"
    error_excerpt: str        # trimmed raw error text (for retrieval + display)
    fix_plan: str             # the successful debug node's plan
    fix_method: str = ""      # code summary of the fix, when available
    domain: str = ""
    timestamp: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BugBookEntry":
        return cls(
            entry_id=data.get("entry_id", ""),
            task_id=data.get("task_id", ""),
            run_id=data.get("run_id", ""),
            error_signature=data.get("error_signature", ""),
            error_excerpt=data.get("error_excerpt", ""),
            fix_plan=data.get("fix_plan", ""),
            fix_method=data.get("fix_method", ""),
            domain=data.get("domain", ""),
            timestamp=data.get("timestamp"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, "")}

    def search_text(self) -> str:
        return f"{self.error_signature}\n{self.error_excerpt}"


@dataclass
class SolutionEntry:
    solution_id: str          # "{task_id}/{run_id}/top{rank}"
    task_id: str
    run_id: str
    rank: int
    metric_value: Optional[float]
    metric_maximize: Optional[bool]
    code_file: str            # relative path inside the store's solutions/ dir
    domain: str = ""
    metric_name: str = ""
    code_head: str = ""       # first lines of the solution, for retrieval indexing
    timestamp: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SolutionEntry":
        return cls(
            solution_id=data.get("solution_id", ""),
            task_id=data.get("task_id", ""),
            run_id=data.get("run_id", ""),
            rank=data.get("rank", 0),
            metric_value=data.get("metric_value"),
            metric_maximize=data.get("metric_maximize"),
            code_file=data.get("code_file", ""),
            domain=data.get("domain", ""),
            metric_name=data.get("metric_name", ""),
            code_head=data.get("code_head", ""),
            timestamp=data.get("timestamp"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, "")}

    def search_text(self) -> str:
        return f"{self.domain}\n{self.metric_name}\n{self.task_id}\n{self.code_head}"
