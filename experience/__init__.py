"""Cross-run experience layer (self-learning autoresearcher).

M1: episodic memory persisted across runs. Runs read a pinned store snapshot;
ingestion from completed runs happens offline via `python -m experience.ingest`.
See docs/self_learning_autoresearcher_plan.md.
"""

from .record import ExperienceRecord
from .store import ExperienceStore

__all__ = ["ExperienceRecord", "ExperienceStore"]
