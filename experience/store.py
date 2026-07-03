"""Persistent cross-run experience store.

Mechanisms (independently switchable for the E4 mechanism ablation):
  M1 episodic  — node-level records (records.jsonl), retrieved into draft/improve.
  M2 lessons   — distilled claims from past runs (lessons.jsonl), same injection.
  M2 bug book  — error-signature -> fix entries (bugbook.jsonl), injected into debug.
  M3 solutions — top solutions from past runs (solutions.jsonl + solutions/*.py),
                 injected into draft as a reference pipeline.

Storage is append-only JSONL in a directory that outlives any single run. A run
opens the store read-only; ingestion happens offline (`python -m experience.ingest`,
`python -m experience.reflect`) so the store contents are a pinned, hashable input
to a run — the snapshot hash covers every store file and is logged with every
injection.

Leakage guard: entries from the current competition, or from any task id in
`excluded_task_ids` (e.g. the task's own fold), are dropped at index build time
across ALL mechanisms — they can never be retrieved.

Retrieval: the episodic corpus uses the existing hybrid BM25+vector retriever when
an embedding model is configured (dependency-free built-in BM25 otherwise); the
small lessons/bugbook/solutions corpora always use the built-in BM25.
"""

import hashlib
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .entries import BugBookEntry, Lesson, SolutionEntry
from .record import ExperienceRecord

logger = logging.getLogger("MLEvolve")

RECORDS_FILENAME = "records.jsonl"
LESSONS_FILENAME = "lessons.jsonl"
BUGBOOK_FILENAME = "bugbook.jsonl"
SOLUTIONS_FILENAME = "solutions.jsonl"
SOLUTIONS_DIRNAME = "solutions"


class _Bm25Only:
    """Minimal Okapi BM25 with the same search() shape as HybridRetriever."""

    K1 = 1.5
    B = 0.75

    def __init__(self, records: List[Any], texts: List[str]):
        self.records = records
        self.corpus = [t.lower().split() for t in texts]
        self.doc_lens = [len(d) for d in self.corpus]
        self.avg_len = (sum(self.doc_lens) / len(self.doc_lens)) if self.corpus else 0.0
        self.doc_freqs: Dict[str, int] = {}
        for doc in self.corpus:
            for term in set(doc):
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

    def search(self, query: str, top_k: int = 10, **_: Any) -> List[Tuple[Any, float]]:
        if not self.corpus:
            return []
        n = len(self.corpus)
        terms = query.lower().split()
        scores = [0.0] * n
        for term in terms:
            df = self.doc_freqs.get(term)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i, doc in enumerate(self.corpus):
                tf = doc.count(term)
                if tf == 0:
                    continue
                denom = tf + self.K1 * (1 - self.B + self.B * self.doc_lens[i] / self.avg_len)
                scores[i] += idf * tf * (self.K1 + 1) / denom
        ranked = sorted(zip(self.records, scores), key=lambda x: x[1], reverse=True)
        return [(r, s) for r, s in ranked if s > 0][:top_k]


class ExperienceStore:
    """Read-only view over the persistent experience corpus for one run."""

    def __init__(
        self,
        store_dir: str,
        current_task_id: str,
        embedding_model_path: str = "",
        embedding_device: str = "cpu",
        excluded_task_ids: Optional[List[str]] = None,
        injection_log_path: Optional[Path] = None,
        top_k: int = 2,
        min_score: float = 0.0,
        use_episodic: bool = True,
        use_lessons: bool = True,
        use_bugbook: bool = True,
        use_solutions: bool = True,
    ):
        if not store_dir:
            raise ValueError("experience.store_dir must be set when the experience store is enabled")
        self.store_dir = Path(store_dir)
        self.records_file = self.store_dir / RECORDS_FILENAME
        self.current_task_id = (current_task_id or "").strip().lower()
        self.excluded_task_ids = {t.strip().lower() for t in (excluded_task_ids or []) if t}
        self.injection_log_path = Path(injection_log_path) if injection_log_path else None
        self.top_k = top_k
        self.min_score = min_score
        self.use_episodic = use_episodic
        self.use_lessons = use_lessons
        self.use_bugbook = use_bugbook
        self.use_solutions = use_solutions

        self._snapshot_hash = compute_snapshot_hash(self.store_dir)
        self.excluded_same_task = 0
        self.excluded_listed = 0

        self.records: List[ExperienceRecord] = self._guarded(
            load_entries(self.records_file, ExperienceRecord) if use_episodic else []
        )
        self.lessons: List[Lesson] = self._guarded(
            load_entries(self.store_dir / LESSONS_FILENAME, Lesson) if use_lessons else []
        )
        self.bugbook: List[BugBookEntry] = self._guarded(
            load_entries(self.store_dir / BUGBOOK_FILENAME, BugBookEntry) if use_bugbook else []
        )
        self.solutions: List[SolutionEntry] = self._guarded(
            load_entries(self.store_dir / SOLUTIONS_FILENAME, SolutionEntry) if use_solutions else []
        )

        self.retriever = self._build_episodic_retriever(embedding_model_path, embedding_device)
        self.lesson_retriever = _Bm25Only(self.lessons, [l.search_text() for l in self.lessons])
        self.bug_retriever = _Bm25Only(self.bugbook, [b.search_text() for b in self.bugbook])
        self.solution_retriever = _Bm25Only(self.solutions, [s.search_text() for s in self.solutions])

        logger.info(
            f"[Experience] Store loaded: {len(self.records)} records, {len(self.lessons)} lessons, "
            f"{len(self.bugbook)} bugbook entries, {len(self.solutions)} solutions retrievable "
            f"({self.excluded_same_task} excluded as same-task '{self.current_task_id}', "
            f"{self.excluded_listed} excluded by list), snapshot={self._snapshot_hash}"
        )

    def _guarded(self, entries: List[Any]) -> List[Any]:
        """Apply the leakage guard: same-task and listed task ids never enter any index."""
        kept = []
        for e in entries:
            tid = (getattr(e, "task_id", "") or "").strip().lower()
            if tid and tid == self.current_task_id:
                self.excluded_same_task += 1
            elif tid in self.excluded_task_ids:
                self.excluded_listed += 1
            else:
                kept.append(e)
        return kept

    def _build_episodic_retriever(self, embedding_model_path: str, embedding_device: str):
        texts = [r.search_text() for r in self.records]
        if not self.records:
            return _Bm25Only([], [])
        if embedding_model_path:
            try:
                from agents.memory.embedding_models import EmbeddingModel
                from agents.memory.retriever import HybridRetriever

                embedding_model = EmbeddingModel(
                    model_type="local", model_name=embedding_model_path, device=embedding_device,
                )
                retriever = HybridRetriever(embedding_model)
                retriever.build_index(self.records, texts)
                return retriever
            except Exception as e:
                logger.warning(f"[Experience] Embedding retriever unavailable ({e}); falling back to BM25-only")
        return _Bm25Only(self.records, texts)

    def snapshot_hash(self) -> str:
        return self._snapshot_hash

    # ---------------- M1 episodic ----------------

    def retrieve(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        label_filter: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> List[Tuple[ExperienceRecord, float]]:
        if not self.records:
            return []
        top_k = top_k if top_k is not None else self.top_k
        min_score = min_score if min_score is not None else self.min_score
        results = self.retriever.search(query_text, top_k=len(self.records), alpha=0.5)
        if label_filter is not None:
            results = [(r, s) for r, s in results if r.label == label_filter]
        results = [(r, s) for r, s in results if s >= min_score]
        return results[:top_k]

    # ---------------- guidance builders ----------------

    def generate_guidance_prompt(self, query_text: str, context_label: str = "") -> str:
        """Cross-run guidance block (episodic + lessons) for draft/improve prompts."""
        successes = self.retrieve(query_text, label_filter=1) if self.use_episodic else []
        failures = self.retrieve(query_text, label_filter=-1) if self.use_episodic else []
        lessons = (
            [l for l, s in self.lesson_retriever.search(query_text, top_k=self.top_k * 2)]
            if self.use_lessons else []
        )
        self._log_injection(context_label, query_text, successes + failures, lessons=lessons)
        if not successes and not failures and not lessons:
            return ""

        parts = [
            "## Cross-Run Experience (from OTHER competitions solved previously)",
            "",
            "These records come from different competitions. Raw metric values are not",
            "comparable to this task — transfer the strategies and pitfalls, not the numbers.",
            "",
        ]
        if lessons:
            parts.append("**📚 Distilled lessons from past runs:**")
            for idx, lesson in enumerate(lessons, 1):
                tags = f" [{', '.join(lesson.scope_tags)}]" if lesson.scope_tags else ""
                conf = f" (confidence: {lesson.confidence})" if lesson.confidence else ""
                parts.append(f"{idx}. {lesson.text}{tags}{conf}")
            parts.append("")
        if successes:
            parts.append("**✅ Approaches that worked on similar problems:**")
            parts.extend(self._format_records(successes))
        if failures:
            parts.append("**❌ Approaches that failed on similar problems (avoid repeating):**")
            parts.extend(self._format_records(failures))
        return "\n".join(parts)

    def augment_memory_text(self, memory_text: str, query_text: str, context_label: str = "") -> str:
        """Append cross-run guidance to an existing Memory prompt section."""
        guidance = self.generate_guidance_prompt(query_text, context_label=context_label)
        if not guidance:
            return memory_text
        memory_text = memory_text or ""
        sep = "\n\n" if memory_text.strip() else ""
        return f"{memory_text}{sep}{guidance}\n"

    def get_bugbook_guidance(self, error_text: str, context_label: str = "debug") -> str:
        """Cross-run fixes for the current error: exact signature match, then BM25."""
        if not self.use_bugbook or not self.bugbook or not (error_text or "").strip():
            return ""
        from .mining import error_signature

        signature = error_signature(error_text)
        exact = [b for b in self.bugbook if signature and b.error_signature == signature]
        matches = [(b, 1.0) for b in exact[: self.top_k]]
        if not matches:
            matches = self.bug_retriever.search(error_text[-2000:], top_k=self.top_k)
        self._log_injection(context_label, error_text, matches)
        if not matches:
            return ""

        parts = [
            "The same or a similar error was fixed in previous runs on OTHER competitions:",
            "",
        ]
        for idx, (bug, _score) in enumerate(matches, 1):
            parts.append(f"{idx}. Error pattern: `{bug.error_signature}`")
            parts.append(f"   Fix that worked: {bug.fix_plan}")
            if bug.fix_method:
                parts.append(f"   Implementation note: {bug.fix_method}")
        parts.append("")
        parts.append("Adapt the fix to the current code — do not copy it blindly.")
        return "\n".join(parts)

    def get_solution_guidance(self, query_text: str, context_label: str = "draft", max_chars: int = 6000) -> str:
        """Reference pipeline from the most similar past competition (top-1)."""
        if not self.use_solutions or not self.solutions:
            return ""
        matches = self.solution_retriever.search(query_text, top_k=1)
        self._log_injection(context_label, query_text, matches)
        if not matches:
            return ""
        sol = matches[0][0]
        code_path = self.store_dir / SOLUTIONS_DIRNAME / sol.code_file
        try:
            code = code_path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            logger.warning(f"[Experience] Solution file unreadable ({code_path}): {e}")
            return ""
        truncated = len(code) > max_chars
        code = code[:max_chars]
        source = f"{sol.domain or sol.task_id}" + (f", {sol.metric_name}" if sol.metric_name else "")
        return "\n".join([
            f"A proven pipeline from a DIFFERENT past competition ({source}, rank top{sol.rank}).",
            "It solves a different dataset: use it as a structural reference for pipeline",
            "organization and technique choices — you MUST adapt data loading, features,",
            "and the target to THIS task, not copy it.",
            "",
            "```python",
            code + ("\n# ... [truncated]" if truncated else ""),
            "```",
        ])

    # ---------------- internals ----------------

    def _format_records(self, results: List[Tuple[ExperienceRecord, float]]) -> List[str]:
        lines = []
        for idx, (rec, _score) in enumerate(results, 1):
            source = rec.domain or rec.task_id
            outcome = {1: "✅ improved/succeeded", -1: "❌ worsened/failed", 0: "⚪ neutral"}[rec.label]
            if rec.metric_name:
                outcome += f" ({rec.metric_name}, {rec.metric_direction or 'unknown direction'})"
            lines.append(f"{idx}. [{source} · {rec.stage}] {outcome}")
            if rec.description:
                lines.append(f"   Plan: {rec.description}")
            if rec.method:
                lines.append(f"   Method: {rec.method}")
        lines.append("")
        return lines

    @staticmethod
    def _entry_id(entry: Any) -> str:
        for attr in ("record_id", "lesson_id", "entry_id", "solution_id"):
            value = getattr(entry, attr, None)
            if value:
                return str(value)
        return ""

    def _log_injection(
        self,
        context_label: str,
        query_text: str,
        results: List[Tuple[Any, float]],
        lessons: Optional[List[Lesson]] = None,
    ) -> None:
        if self.injection_log_path is None:
            return
        retrieved = [
            {"id": self._entry_id(e), "task_id": getattr(e, "task_id", ""), "score": round(float(s), 6)}
            for e, s in results
        ]
        retrieved += [
            {"id": l.lesson_id, "task_id": l.task_id, "score": None} for l in (lessons or [])
        ]
        entry = {
            "timestamp": datetime.now().isoformat(),
            "context": context_label,
            "query_chars": len(query_text),
            "snapshot_hash": self._snapshot_hash,
            "current_task_id": self.current_task_id,
            "excluded_same_task": self.excluded_same_task,
            "excluded_listed": self.excluded_listed,
            "retrieved": retrieved,
        }
        try:
            self.injection_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.injection_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"[Experience] Failed to write injection log: {e}")


# ---------------- module-level store IO ----------------

def load_entries(path: Path, cls) -> List[Any]:
    entries: List[Any] = []
    if not path.exists():
        return entries
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(cls.from_dict(json.loads(line)))
            except Exception as e:
                logger.warning(f"[Experience] Skipping malformed entry at {path}:{line_no}: {e}")
    return entries


def load_records(records_file: Path) -> List[ExperienceRecord]:
    return load_entries(records_file, ExperienceRecord)


def append_entries(path: Path, entries: List[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")


def append_records(records_file: Path, records: List[ExperienceRecord]) -> None:
    append_entries(records_file, records)


def compute_snapshot_hash(store_path: Path) -> str:
    """Content hash identifying the exact store state a run consumed.

    Accepts the store directory (hashes every store file, sorted by relative
    path) or a single file (back-compatible with early callers).
    """
    h = hashlib.sha256()
    store_path = Path(store_path)
    if store_path.is_dir():
        files = sorted(
            p for p in store_path.rglob("*")
            if p.is_file() and (p.suffix in (".jsonl", ".py")) and not p.name.startswith(".")
        )
        for p in files:
            h.update(str(p.relative_to(store_path)).encode())
            h.update(p.read_bytes())
    elif store_path.exists():
        h.update(store_path.read_bytes())
    return h.hexdigest()[:16]
