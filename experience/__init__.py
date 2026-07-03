"""Cross-run experience layer (self-learning autoresearcher).

M1 episodic memory (records), M2 lessons + bug book, M3 solution library —
persisted across runs, read-only during a run, ingested offline via
`python -m experience.ingest` (deterministic) and `python -m experience.reflect`
(LLM lesson distillation). See docs/self_learning_autoresearcher_plan.md.
"""

from .entries import BugBookEntry, Lesson, SolutionEntry
from .record import ExperienceRecord
from .store import ExperienceStore

__all__ = ["BugBookEntry", "ExperienceRecord", "ExperienceStore", "Lesson", "SolutionEntry"]
