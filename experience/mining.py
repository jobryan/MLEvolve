"""Deterministic miners for M2 (bug book) and M3 (solution library).

No LLM calls here — everything is extracted mechanically from run artifacts:
  mine_bugbook()   — walk logs/journal.json debug chains: buggy parent -> fixed child.
  index_solutions() — copy workspace/top_solution/top{k} code into the store and index it.
LLM-based lesson distillation lives in experience/reflect.py.
"""

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .entries import BugBookEntry, SolutionEntry

logger = logging.getLogger("MLEvolve")

_ERROR_LINE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Interrupt|Warning))\b\s*:?\s*(.*)")


def error_signature(error_text: str, exc_type: str = "") -> str:
    """Normalize an error into a stable signature: 'ErrType: message-shape'.

    Numbers, hex ids, quoted values, and paths are collapsed so the same root
    cause matches across tasks and runs.
    """
    etype, message = exc_type or "", ""
    for line in reversed((error_text or "").strip().splitlines()):
        m = _ERROR_LINE.search(line.strip())
        if m:
            etype = etype or m.group(1)
            message = m.group(2) or message
            if m.group(1) and m.group(2):
                etype, message = m.group(1), m.group(2)
                break
    if not etype:
        return ""
    message = re.sub(r"'[^']*'|\"[^\"]*\"", "'<val>'", message)
    message = re.sub(r"/[^\s,;)]+", "<path>", message)
    message = re.sub(r"0x[0-9a-fA-F]+", "<hex>", message)
    message = re.sub(r"\d+", "<n>", message)
    message = re.sub(r"\s+", " ", message).strip()
    return f"{etype}: {message}"[:200]


def _node_error_text(node: Dict[str, Any]) -> str:
    term_out = node.get("_term_out") or []
    if isinstance(term_out, list):
        text = "".join(str(t) for t in term_out)
    else:
        text = str(term_out)
    return text[-4000:]


def load_journal(journal_path: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """Return (node_id -> node dict, node_id -> parent_id) from a serialized journal."""
    with open(journal_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    nodes = {n["id"]: n for n in data.get("nodes", []) if n.get("id")}
    return nodes, data.get("node2parent", {}) or {}


def mine_bugbook(
    journal_path: Path,
    task_id: str,
    run_id: str,
    domain: str = "",
) -> List[BugBookEntry]:
    """Extract error->fix pairs from successful debug chains in a run journal."""
    nodes, node2parent = load_journal(journal_path)
    entries: List[BugBookEntry] = []
    seen_signatures = set()

    for node in nodes.values():
        if node.get("stage") != "debug" or node.get("is_buggy") is not False:
            continue
        parent = nodes.get(node2parent.get(node.get("id", ""), ""))
        if not parent or parent.get("is_buggy") is not True:
            continue

        error_text = _node_error_text(parent)
        signature = error_signature(error_text, parent.get("exc_type") or "")
        fix_plan = (node.get("plan") or "").strip()
        if not signature or not fix_plan:
            continue
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        entries.append(
            BugBookEntry(
                entry_id=f"{task_id}/{run_id}/bug_{len(entries)}",
                task_id=task_id,
                run_id=run_id,
                error_signature=signature,
                error_excerpt=error_text[-1500:],
                fix_plan=fix_plan[:2000],
                fix_method=(node.get("code_summary") or "")[:1000],
                domain=domain,
                timestamp=node.get("created_time") or None,
            )
        )
    return entries


def mine_episodic_records(
    journal_path: Path,
    task_id: str,
    run_id: str,
    domain: str = "",
    metric_name: str = "",
    metric_direction: str = "",
) -> List["ExperienceRecord"]:
    """Fallback episodic ingestion straight from logs/journal.json.

    Used when a run has no global_memory/records.json (e.g. the memory layer
    silently disabled on a CPU/offline worker). Mirrors GlobalMemoryLayer's
    save rules: only non-buggy nodes with a metric value; labels derived from
    stage and parent-metric comparison. Record ids use a 'jnode_' prefix so a
    later records.json ingest of the same run cannot double-count nodes.
    """
    from .record import ExperienceRecord

    nodes, node2parent = load_journal(journal_path)
    records: List[ExperienceRecord] = []
    for node in nodes.values():
        if node.get("is_buggy") is not False:
            continue
        metric = node.get("metric") or {}
        value = metric.get("value") if isinstance(metric, dict) else None
        if value is None:
            continue
        parent = nodes.get(node2parent.get(node.get("id", ""), ""))
        stage = node.get("stage") or "unknown"

        label = 0
        if stage in ("draft", "fusion_draft"):
            label = 1
        elif stage == "debug":
            label = 1 if (parent and parent.get("is_buggy") is True) else 0
        elif stage in ("improve", "evolution", "fusion") and parent:
            parent_metric = (parent.get("metric") or {}).get("value") if isinstance(parent.get("metric"), dict) else None
            if parent_metric is not None:
                maximize = metric.get("maximize")
                if maximize is None:
                    maximize = metric_direction != "minimize"
                if value != parent_metric:
                    label = 1 if ((value > parent_metric) == bool(maximize)) else -1

        plan = (node.get("plan") or "").strip()
        method = (node.get("code_summary") or "").strip() or plan[:500]
        if not plan and not method:
            continue

        parent_error = ""
        if stage == "debug" and parent:
            parent_error = _node_error_text(parent)[-1500:]

        parent_metric_value = None
        if parent and isinstance(parent.get("metric"), dict):
            parent_metric_value = parent["metric"].get("value")

        records.append(
            ExperienceRecord(
                record_id=f"{task_id}/{run_id}/jnode_{node.get('id')}",
                task_id=task_id,
                run_id=run_id,
                stage=stage,
                description=plan[:2000],
                method=method[:1000],
                label=label,
                domain=domain,
                metric_name=metric_name,
                metric_direction=metric_direction,
                parent_metric=parent_metric_value,
                current_metric=value,
                exec_time=node.get("exec_time"),
                parent_error=parent_error,
                timestamp=node.get("created_time") or None,
            )
        )
    return records


def _parse_metric_file(metric_file: Path) -> Tuple[Optional[float], Optional[bool]]:
    value, maximize = None, None
    try:
        for line in metric_file.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("Metric:"):
                try:
                    value = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("Maximize:"):
                maximize = line.split(":", 1)[1].strip().lower() == "true"
    except OSError:
        pass
    return value, maximize


def index_solutions(
    run_dir: Path,
    store_dir: Path,
    task_id: str,
    run_id: str,
    domain: str = "",
    metric_name: str = "",
    top_n: int = 3,
) -> List[SolutionEntry]:
    """Copy the run's top-N solutions into the store and return index entries."""
    workspace = run_dir / "workspace" if (run_dir / "workspace").is_dir() else run_dir
    top_dir = workspace / "top_solution"
    candidates = []
    if top_dir.is_dir():
        for rank in range(1, top_n + 1):
            rank_dir = top_dir / f"top{rank}"
            if (rank_dir / "solution.py").exists():
                candidates.append((rank, rank_dir))
    elif (workspace / "best_solution" / "solution.py").exists():
        candidates.append((1, workspace / "best_solution"))

    solutions_dir = store_dir / "solutions"
    entries: List[SolutionEntry] = []
    for rank, rank_dir in candidates:
        code_path = rank_dir / "solution.py"
        code = code_path.read_text(encoding="utf-8", errors="replace")
        metric_value, metric_maximize = _parse_metric_file(rank_dir / "metric.txt")

        solution_id = f"{task_id}/{run_id}/top{rank}"
        rel_file = f"{solution_id.replace('/', '__')}.py"
        solutions_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(code_path, solutions_dir / rel_file)

        entries.append(
            SolutionEntry(
                solution_id=solution_id,
                task_id=task_id,
                run_id=run_id,
                rank=rank,
                metric_value=metric_value,
                metric_maximize=metric_maximize,
                code_file=rel_file,
                domain=domain,
                metric_name=metric_name,
                code_head=code[:1500],
            )
        )
    return entries
