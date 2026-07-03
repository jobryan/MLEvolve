"""Persistent cross-run experience store (M1: episodic memory).

Storage: append-only JSONL (`records.jsonl`) in a directory that outlives any
single run. A run opens the store read-only; ingestion happens offline via
`python -m experience.ingest` so the store contents are a pinned, hashable
input to a run (the snapshot hash is logged with every injection).

Leakage guard: records from the current competition, or from any task id in
`excluded_task_ids` (e.g. a held-out evaluation split), are dropped at index
build time — they can never be retrieved.

Retrieval: hybrid BM25+vector via the existing agents.memory retriever when an
embedding model path is configured; otherwise a dependency-free built-in BM25.
"""

import hashlib
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .record import ExperienceRecord

logger = logging.getLogger("MLEvolve")

RECORDS_FILENAME = "records.jsonl"


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

        all_records = load_records(self.records_file)
        self._snapshot_hash = compute_snapshot_hash(self.records_file)

        # Leakage guard applied at index build time: guarded records never enter the index.
        self.records: List[ExperienceRecord] = []
        self.excluded_same_task = 0
        self.excluded_listed = 0
        for rec in all_records:
            tid = rec.task_id.strip().lower()
            if tid and tid == self.current_task_id:
                self.excluded_same_task += 1
            elif tid in self.excluded_task_ids:
                self.excluded_listed += 1
            else:
                self.records.append(rec)

        texts = [r.search_text() for r in self.records]
        self.retriever = self._build_retriever(texts, embedding_model_path, embedding_device)
        logger.info(
            f"[Experience] Store loaded: {len(self.records)} retrievable records "
            f"({self.excluded_same_task} excluded as same-task '{self.current_task_id}', "
            f"{self.excluded_listed} excluded by list), snapshot={self._snapshot_hash}"
        )

    def _build_retriever(self, texts: List[str], embedding_model_path: str, embedding_device: str):
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

    def generate_guidance_prompt(self, query_text: str, context_label: str = "") -> str:
        """Cross-run guidance block for prompts; logs the injection with provenance."""
        successes = self.retrieve(query_text, label_filter=1)
        failures = self.retrieve(query_text, label_filter=-1)
        self._log_injection(context_label, query_text, successes + failures)
        if not successes and not failures:
            return ""

        parts = [
            "## Cross-Run Experience (from OTHER competitions solved previously)",
            "",
            "These records come from different competitions. Raw metric values are not",
            "comparable to this task — transfer the strategies and pitfalls, not the numbers.",
            "",
        ]
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

    def _log_injection(
        self, context_label: str, query_text: str, results: List[Tuple[ExperienceRecord, float]]
    ) -> None:
        if self.injection_log_path is None:
            return
        entry = {
            "timestamp": datetime.now().isoformat(),
            "context": context_label,
            "query_chars": len(query_text),
            "snapshot_hash": self._snapshot_hash,
            "current_task_id": self.current_task_id,
            "excluded_same_task": self.excluded_same_task,
            "excluded_listed": self.excluded_listed,
            "retrieved": [
                {"record_id": r.record_id, "task_id": r.task_id, "stage": r.stage,
                 "label": r.label, "score": round(float(s), 6)}
                for r, s in results
            ],
        }
        try:
            self.injection_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.injection_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"[Experience] Failed to write injection log: {e}")


def load_records(records_file: Path) -> List[ExperienceRecord]:
    records: List[ExperienceRecord] = []
    if not records_file.exists():
        return records
    with open(records_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(ExperienceRecord.from_dict(json.loads(line)))
            except Exception as e:
                logger.warning(f"[Experience] Skipping malformed record at {records_file}:{line_no}: {e}")
    return records


def append_records(records_file: Path, records: List[ExperienceRecord]) -> None:
    records_file.parent.mkdir(parents=True, exist_ok=True)
    with open(records_file, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")


def compute_snapshot_hash(records_file: Path) -> str:
    """Content hash identifying the exact store state a run consumed."""
    h = hashlib.sha256()
    if records_file.exists():
        with open(records_file, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()[:16]
