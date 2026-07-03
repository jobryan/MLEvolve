"""Offline LLM reflection: distill a completed run's journal into lessons (M2).

Usage:
    python -m experience.reflect --store-dir /path/to/store RUN_DIR [RUN_DIR ...] \
        [--task-id ID] [--run-id RID] [--domain D] \
        [--model gpt-4.1] [--base-url URL] [--api-key KEY] [--temp 0.3] \
        [--max-lessons 6] [--dry-run]

Reads logs/journal.json, summarizes what was tried and what happened, and asks
an OpenAI-compatible model for a handful of generalizable lessons with scope
tags and self-assessed confidence. Lessons are appended to lessons.jsonl with
full provenance. Re-running on an already-reflected (task, run) is a no-op.

API key resolution: --api-key, else $OPENAI_API_KEY.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from .entries import Lesson
from .ingest import derive_task_id, read_run_config
from .mining import error_signature, load_journal
from .store import LESSONS_FILENAME, append_entries, compute_snapshot_hash, load_entries

MAX_PLAN_CHARS = 300
MAX_NODES_IN_SUMMARY = 30

LESSON_SCHEMA = {
    "type": "object",
    "properties": {
        "lessons": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "scope_tags": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": ["text", "scope_tags", "confidence"],
            },
        }
    },
    "required": ["lessons"],
}


def summarize_journal(journal_path: Path) -> str:
    """Compact, LLM-readable digest of a run's search tree."""
    nodes, node2parent = load_journal(journal_path)
    valid = [n for n in nodes.values() if n.get("is_buggy") is False and _metric_value(n) is not None]
    buggy = [n for n in nodes.values() if n.get("is_buggy") is True]
    valid.sort(key=lambda n: n.get("step") or 0)

    lines = [
        f"Run tree: {len(nodes)} nodes total, {len(valid)} valid/scored, {len(buggy)} buggy.",
        "",
        "Chronological valid nodes (stage | node_id[:8] | metric | improved-vs-parent | plan excerpt):",
    ]
    for n in valid[:MAX_NODES_IN_SUMMARY]:
        parent = nodes.get(node2parent.get(n.get("id", ""), ""))
        parent_metric = _metric_value(parent) if parent else None
        metric = _metric_value(n)
        improved = "n/a"
        if parent_metric is not None and metric is not None:
            improved = "yes" if metric != parent_metric else "no-change"
        plan = re.sub(r"\s+", " ", (n.get("plan") or ""))[:MAX_PLAN_CHARS]
        lines.append(f"- {n.get('stage')} | {str(n.get('id'))[:8]} | {metric} | {improved} | {plan}")
    if len(valid) > MAX_NODES_IN_SUMMARY:
        lines.append(f"- ... ({len(valid) - MAX_NODES_IN_SUMMARY} more valid nodes omitted)")

    signatures: Dict[str, int] = {}
    for n in buggy:
        sig = error_signature("".join(str(t) for t in (n.get("_term_out") or []))[-4000:], n.get("exc_type") or "")
        if sig:
            signatures[sig] = signatures.get(sig, 0) + 1
    if signatures:
        lines.append("")
        lines.append("Recurring errors (signature: count):")
        for sig, count in sorted(signatures.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"- {sig}: {count}")

    return "\n".join(lines)


def _metric_value(node: Optional[Dict[str, Any]]) -> Optional[float]:
    if not node:
        return None
    metric = node.get("metric")
    if isinstance(metric, dict):
        return metric.get("value")
    return None


def build_reflection_prompt(task_id: str, domain: str, journal_summary: str, max_lessons: int) -> str:
    return f"""You are analyzing the complete experiment log of an autonomous ML-engineering agent that just finished a Kaggle-style competition (task: {task_id}{f', domain: {domain}' if domain else ''}).

{journal_summary}

Distill at most {max_lessons} LESSONS this agent should carry into FUTURE, DIFFERENT competitions.

Rules for a good lesson:
- Generalizable: about strategy, technique choice, budget allocation, or pitfalls — NOT about this dataset's specifics, and NEVER citing this task's metric numbers.
- Self-contained and actionable: a future agent reading only the lesson text must know what to do differently.
- Evidence-based: only claim what this log actually supports; mark speculative extrapolations as low confidence.
- scope_tags: 2-4 lowercase tags for when the lesson applies (e.g. "tabular", "small-dataset", "image-classification", "debugging", "ensembling", "time-budget").

Return JSON: {{"lessons": [{{"text": ..., "scope_tags": [...], "confidence": "high|medium|low"}}]}}"""


def call_llm(prompt: str, model: str, base_url: str, api_key: str, temp: float) -> List[Dict[str, Any]]:
    from llm import generate  # repo backend; requires repo deps installed

    stage = SimpleNamespace(model=model, temp=temp, base_url=base_url, api_key=api_key)
    cfg = SimpleNamespace(agent=SimpleNamespace(code=stage, feedback=stage))
    response = generate(prompt=prompt, cfg=cfg, temperature=temp, json_schema=LESSON_SCHEMA)

    text = response if isinstance(response, str) else json.dumps(response)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    parsed = json.loads(text) if isinstance(response, str) else response
    lessons = parsed.get("lessons", []) if isinstance(parsed, dict) else parsed
    if not isinstance(lessons, list):
        raise ValueError(f"unexpected lessons payload: {type(lessons)}")
    return lessons


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--store-dir", required=True, type=Path)
    parser.add_argument("--task-id", default="", help="Competition id (single run dir only)")
    parser.add_argument("--run-id", default="", help="Run id override (single run dir only)")
    parser.add_argument("--domain", default="")
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--temp", type=float, default=0.3)
    parser.add_argument("--max-lessons", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true", help="Print the prompt and skip the LLM call")
    args = parser.parse_args(argv)

    if len(args.run_dirs) > 1 and (args.task_id or args.run_id):
        parser.error("--task-id/--run-id may only be used with a single RUN_DIR")
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key and not args.dry_run:
        parser.error("no API key: pass --api-key or set OPENAI_API_KEY")

    lessons_file = args.store_dir / LESSONS_FILENAME
    existing = load_entries(lessons_file, Lesson)
    reflected = {(l.task_id, l.run_id) for l in existing}

    total = 0
    for run_dir in args.run_dirs:
        journal_path = next(
            (p for p in (run_dir / "logs" / "journal.json", run_dir / "journal.json") if p.exists()), None,
        )
        if journal_path is None:
            print(f"[skip] {run_dir}: no logs/journal.json", file=sys.stderr)
            continue
        run_config = read_run_config(run_dir)
        task_id = (args.task_id or derive_task_id(run_config)).strip().lower()
        run_id = args.run_id or run_config.get("exp_name") or run_dir.name
        if not task_id:
            print(f"[skip] {run_dir}: task id underivable; pass --task-id", file=sys.stderr)
            continue
        if (task_id, run_id) in reflected:
            print(f"[skip] {run_dir}: already reflected (task={task_id} run={run_id})")
            continue

        summary = summarize_journal(journal_path)
        prompt = build_reflection_prompt(task_id, args.domain, summary, args.max_lessons)
        if args.dry_run:
            print(f"--- prompt for {run_dir} (task={task_id} run={run_id}) ---\n{prompt}\n")
            continue

        raw_lessons = call_llm(prompt, args.model, args.base_url, api_key, args.temp)[: args.max_lessons]
        lessons = [
            Lesson(
                lesson_id=f"{task_id}/{run_id}/lesson_{i}",
                task_id=task_id,
                run_id=run_id,
                text=str(l.get("text", "")).strip(),
                scope_tags=[str(t).lower() for t in (l.get("scope_tags") or [])][:4],
                confidence=str(l.get("confidence", "")).lower(),
                domain=args.domain,
                timestamp=datetime.now().isoformat(),
            )
            for i, l in enumerate(raw_lessons)
            if str(l.get("text", "")).strip()
        ]
        append_entries(lessons_file, lessons)
        reflected.add((task_id, run_id))
        total += len(lessons)
        print(f"[ok] {run_dir}: task={task_id} run={run_id} lessons={len(lessons)}")

    print(f"Done: added {total} lessons")
    print(f"Store snapshot: {compute_snapshot_hash(args.store_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
