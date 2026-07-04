"""Build a placebo experience store (E1 arm C).

The placebo controls for prompt-token count and for the "having a memory"
scaffolding effect while destroying the informational content of experience:
every corpus keeps its exact size, structure, labels, provenance fields, and
token multiset, but the *content* is deranged (a fixed-point-free permutation)
so retrieved text no longer corresponds to its outcome, provenance, or query.

Per corpus:
  records.jsonl   — (description, method) deranged across records; task/run/stage/
                    label/metric metadata stay in place.
  lessons.jsonl   — text deranged across lessons; scope_tags/confidence/provenance stay.
  bugbook.jsonl   — (fix_plan, fix_method) deranged across entries; signatures stay.
  solutions.jsonl — (code_file, code_head) deranged across entries; index metadata stays.

Deterministic under --seed. Writes placebo_meta.json recording the source store
snapshot, seed, and method so the placebo store is as auditable as the real one.

Usage:
    python -m experience.placebo --source-store REAL_STORE --output-store PLACEBO_STORE [--seed 20260702]
"""

import argparse
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional

from .entries import BugBookEntry, Lesson, SolutionEntry
from .record import ExperienceRecord
from .store import (
    BUGBOOK_FILENAME,
    LESSONS_FILENAME,
    RECORDS_FILENAME,
    SOLUTIONS_DIRNAME,
    SOLUTIONS_FILENAME,
    append_entries,
    compute_snapshot_hash,
    load_entries,
)


def derangement(n: int, rng: random.Random) -> List[int]:
    """Fixed-point-free permutation of range(n), deterministic under rng."""
    if n < 2:
        return list(range(n))
    perm = list(range(n))
    rng.shuffle(perm)
    for i in range(n):
        if perm[i] == i:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]
    assert all(perm[i] != i for i in range(n))
    return perm


def _derange_fields(entries: List[Any], fields: List[str], rng: random.Random) -> int:
    """Permute the given field-tuple across entries with no fixed points."""
    if len(entries) < 2:
        return 0
    perm = derangement(len(entries), rng)
    originals = [{f: getattr(e, f) for f in fields} for e in entries]
    for i, entry in enumerate(entries):
        for f in fields:
            setattr(entry, f, originals[perm[i]][f])
    return len(entries)


def build_placebo(source_store: Path, output_store: Path, seed: int) -> dict:
    if output_store.exists() and any(output_store.iterdir()):
        raise SystemExit(f"output store {output_store} already exists and is not empty")
    source_snapshot = compute_snapshot_hash(source_store)
    rng = random.Random(seed)
    stats = {}

    corpora: List[tuple] = [
        (RECORDS_FILENAME, ExperienceRecord, ["description", "method"]),
        (LESSONS_FILENAME, Lesson, ["text"]),
        (BUGBOOK_FILENAME, BugBookEntry, ["fix_plan", "fix_method"]),
        (SOLUTIONS_FILENAME, SolutionEntry, ["code_file", "code_head"]),
    ]
    for filename, cls, fields in corpora:
        entries = load_entries(source_store / filename, cls)
        deranged = _derange_fields(entries, fields, rng)
        if entries:
            append_entries(output_store / filename, entries)
        stats[filename] = {"entries": len(entries), "deranged": deranged}

    solutions_src = source_store / SOLUTIONS_DIRNAME
    if solutions_src.is_dir():
        shutil.copytree(solutions_src, output_store / SOLUTIONS_DIRNAME)
        stats["solution_files_copied"] = len(list(solutions_src.glob("*.py")))

    meta = {
        "kind": "placebo",
        "method": "fixed-point-free content derangement per corpus",
        "source_store": str(source_store),
        "source_snapshot": source_snapshot,
        "seed": seed,
        "created_at": datetime.now().isoformat(),
        "stats": stats,
    }
    (output_store / "placebo_meta.json").write_text(json.dumps(meta, indent=2))
    meta["placebo_snapshot"] = compute_snapshot_hash(output_store)
    return meta


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-store", required=True, type=Path)
    parser.add_argument("--output-store", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260702)
    args = parser.parse_args(argv)

    meta = build_placebo(args.source_store, args.output_store, args.seed)
    for filename, s in meta["stats"].items():
        if isinstance(s, dict):
            print(f"{filename}: {s['entries']} entries, {s['deranged']} deranged")
        else:
            print(f"{filename}: {s}")
    print(f"Source snapshot:  {meta['source_snapshot']}")
    print(f"Placebo snapshot: {meta['placebo_snapshot']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
