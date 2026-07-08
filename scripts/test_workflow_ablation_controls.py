#!/usr/bin/env python3
"""Dependency-light checks for workflow/operator ablation controls."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.workflow_controls import code_review_enabled, operator_allowed


def make_agent(operator_set="full", use_code_review=True):
    return SimpleNamespace(
        cfg=SimpleNamespace(ablation=SimpleNamespace(operator_set=operator_set)),
        acfg=SimpleNamespace(use_code_review=use_code_review),
    )


def test_aide_operator_set() -> None:
    agent = make_agent(operator_set="draft_debug_improve")
    assert operator_allowed(agent, "Draft")
    assert operator_allowed(agent, "Debug")
    assert operator_allowed(agent, "Improve")
    assert not operator_allowed(agent, "Evolution")
    assert not operator_allowed(agent, "Fusion/Crossover")
    assert not operator_allowed(agent, "Aggregation")


def test_full_operator_set() -> None:
    agent = make_agent(operator_set="full")
    assert operator_allowed(agent, "Evolution")
    assert operator_allowed(agent, "Fusion/Crossover")
    assert operator_allowed(agent, "Aggregation")


def test_no_fusion_evolution_operator_set() -> None:
    agent = make_agent(operator_set="no_fusion_evolution")
    assert operator_allowed(agent, "Draft")
    assert operator_allowed(agent, "Improve")
    assert not operator_allowed(agent, "Evolution")
    assert not operator_allowed(agent, "Fusion/Crossover")


def test_code_review_switch() -> None:
    assert code_review_enabled(make_agent(use_code_review=True))
    assert not code_review_enabled(make_agent(use_code_review=False))


def main() -> int:
    test_aide_operator_set()
    test_full_operator_set()
    test_no_fusion_evolution_operator_set()
    test_code_review_switch()
    print("workflow ablation control tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
