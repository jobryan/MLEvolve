"""Ablation controls for workflow/operator availability."""

from __future__ import annotations

from typing import Any


OPERATOR_SETS = {
    "full": {"Draft", "Debug", "Improve", "Evolution", "Fusion/Crossover", "Aggregation"},
    "draft_debug_improve": {"Draft", "Debug", "Improve"},
    "no_fusion_evolution": {"Draft", "Debug", "Improve", "Aggregation"},
    "draft_only": {"Draft"},
}


def ablation_cfg(agent: Any) -> Any:
    return getattr(getattr(agent, "cfg", None), "ablation", None)


def operator_set(agent: Any) -> str:
    return str(getattr(ablation_cfg(agent), "operator_set", "full") or "full")


def operator_allowed(agent: Any, operator: str) -> bool:
    allowed = OPERATOR_SETS.get(operator_set(agent), OPERATOR_SETS["full"])
    return operator in allowed


def code_review_enabled(agent: Any) -> bool:
    return bool(getattr(getattr(agent, "acfg", None), "use_code_review", True))


def diversity_prompts_enabled(agent: Any) -> bool:
    """diversity_mode='none' removes novelty/diversity instructions from prompts."""
    mode = getattr(ablation_cfg(agent), "diversity_mode", "default")
    return str(mode or "default") != "none"
