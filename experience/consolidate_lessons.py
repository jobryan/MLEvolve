"""Consolidate a store's lessons into a canonical, deduplicated set (F4).

E1 v1 shipped 150-170 per-run lessons per fold with heavy near-duplication, so
retrieval sampled ~4 near-arbitrary ones. This pass merges them into <=40
canonical lessons, each carrying the provenance ids of the raw lessons it
subsumes. The original file is preserved as lessons_raw.jsonl.

Usage:
    python -m experience.consolidate_lessons --store-dir stores/e1_fold_a \
        [--max-lessons 40] [--model gpt-4.1] [--api-key KEY]
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .entries import Lesson
from .reflect import call_llm
from .store import LESSONS_FILENAME, append_entries, compute_snapshot_hash, load_entries

CONSOLIDATE_SCHEMA = {
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
                    "source_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "scope_tags", "confidence", "source_ids"],
            },
        }
    },
    "required": ["lessons"],
}


def build_prompt(lessons: List[Lesson], max_lessons: int) -> str:
    listing = "\n".join(
        f"[{l.lesson_id}] (tags: {','.join(l.scope_tags)}; confidence: {l.confidence}) {l.text}"
        for l in lessons
    )
    return f"""Below are {len(lessons)} lessons distilled independently from many runs of an
autonomous ML-engineering agent. They contain heavy near-duplication.

{listing}

Merge them into AT MOST {max_lessons} canonical lessons. Rules:
- Merge lessons expressing the same advice; keep the clearest, most actionable phrasing.
- A canonical lesson's confidence is "high" only if multiple independent sources support it.
- Preserve genuinely distinct advice even if only one source states it (confidence: low).
- Keep scope_tags: 2-4 lowercase tags per lesson describing when it applies.
- source_ids: list EVERY input lesson id (the [bracketed] ids) each canonical lesson subsumes.
- Do not invent advice not present in the inputs.

Return JSON: {{"lessons": [{{"text": ..., "scope_tags": [...], "confidence": ..., "source_ids": [...]}}]}}"""


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store-dir", required=True, type=Path)
    parser.add_argument("--max-lessons", type=int, default=40)
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--temp", type=float, default=0.2)
    args = parser.parse_args(argv)

    import os

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        parser.error("no API key: pass --api-key or set OPENAI_API_KEY")

    lessons_file = args.store_dir / LESSONS_FILENAME
    raw = load_entries(lessons_file, Lesson)
    if not raw:
        print("no lessons to consolidate")
        return 0

    result = call_llm(build_prompt(raw, args.max_lessons), args.model, args.base_url, api_key, args.temp)

    by_id = {l.lesson_id: l for l in raw}
    canonical = []
    for i, item in enumerate(result[: args.max_lessons]):
        sources = [s for s in (item.get("source_ids") or []) if s in by_id]
        src_tasks = sorted({by_id[s].task_id for s in sources}) if sources else []
        canonical.append(Lesson(
            lesson_id=f"canonical/{args.store_dir.name}/lesson_{i}",
            task_id=src_tasks[0] if len(src_tasks) == 1 else "",
            run_id="consolidated",
            text=str(item.get("text", "")).strip(),
            scope_tags=[str(t).lower() for t in (item.get("scope_tags") or [])][:4],
            confidence=str(item.get("confidence", "")).lower(),
            provenance_node_ids=",".join(sources),
            domain=by_id[sources[0]].domain if sources else "",
            timestamp=datetime.now().isoformat(),
        ))
    canonical = [l for l in canonical if l.text]

    backup = args.store_dir / "lessons_raw.jsonl"
    if not backup.exists():
        shutil.copyfile(lessons_file, backup)
    lessons_file.unlink()
    append_entries(lessons_file, canonical)

    covered = len({s for l in canonical for s in l.provenance_node_ids.split(",") if s})
    print(f"consolidated {len(raw)} -> {len(canonical)} lessons ({covered}/{len(raw)} sources covered)")
    print(f"store snapshot: {compute_snapshot_hash(args.store_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
