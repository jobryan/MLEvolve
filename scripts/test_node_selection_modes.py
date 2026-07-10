#!/usr/bin/env python3
"""Dependency-light checks for MLEvolve search policy selection modes."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class FakeMetric:
    def __init__(self, value: float):
        self.value = value


class FakeNode:
    def __init__(
        self,
        node_id: str,
        *,
        metric: float | None = None,
        reached_limit: bool = False,
        uct: float = 0.0,
        is_buggy: bool | None = False,
        stage: str = "improve",
    ):
        self.id = node_id
        self.metric = FakeMetric(metric) if metric is not None else None
        self._reached_limit = reached_limit
        self._uct = uct
        self.is_buggy = is_buggy
        self.is_terminal = False
        self.stage = stage
        self.plan = ""
        self.code_summary = None
        self.code = ""
        self.children = []
        self.parent = None
        self.lock = False
        self.continue_improve = False
        self.is_debug_success = False
        self.step = 0
        self.ctime = 0.0

    def add_child(self, child: "FakeNode") -> "FakeNode":
        child.parent = self
        self.children.append(child)
        return child

    def reached_child_limit(self, scfg, for_topk: bool = False) -> bool:
        return self._reached_limit

    def uct_value(self, exploration_constant: float = 1.414) -> float:
        return self._uct


class FakeAgent:
    def __init__(self, policy: str, root: FakeNode, nodes: list[FakeNode], novelty_lambda: float = 0.0):
        decay = SimpleNamespace(
            exploration_constant=1.414,
            lower_bound=0.5,
            alpha=0.01,
            phase_ratios=[0.3, 0.7],
        )
        self.cfg = SimpleNamespace(
            ablation=SimpleNamespace(search_policy=policy, novelty_lambda=novelty_lambda),
            agent=SimpleNamespace(decay=decay),
        )
        self.acfg = SimpleNamespace(time_limit=100, steps=10)
        self.scfg = SimpleNamespace(
            num_drafts=2,
            num_improves=2,
            explore_switch_start=0.5,
            explore_switch_end=0.7,
            min_exploration_weight=0.2,
        )
        self.virtual_root = root
        self.branch_all_nodes = {0: nodes}
        self.journal = SimpleNamespace(nodes=nodes)
        self.metric_maximize = True
        self.search_start_time = None
        self.current_step = 0

    def is_root(self, node: FakeNode) -> bool:
        return node is self.virtual_root


def load_node_selection():
    search_node_module = types.ModuleType("engine.search_node")
    search_node_module.SearchNode = FakeNode
    conditions_module = types.ModuleType("engine.conditions")
    conditions_module.should_trigger_branch_fusion = lambda agent: False
    sys.modules["engine.search_node"] = search_node_module
    sys.modules["engine.conditions"] = conditions_module

    spec = importlib.util.spec_from_file_location(
        "node_selection_under_test",
        ROOT / "engine/node_selection.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def assert_selected(name: str, actual: FakeNode, expected: FakeNode) -> None:
    if actual is not expected:
        raise AssertionError(f"{name}: expected {expected.id}, got {actual.id}")


def test_linear_chain(ns) -> None:
    root = FakeNode("root", reached_limit=True, stage="root")
    low = root.add_child(FakeNode("low", metric=0.3, reached_limit=False))
    high = root.add_child(FakeNode("high", metric=0.8, reached_limit=False))
    agent = FakeAgent("linear_chain", root, [low, high])

    selected = ns.select_with_soft_switch(agent)

    assert_selected("linear_chain", selected, high)
    assert agent.last_selection_rationale["policy"] == "linear_chain"


def test_greedy_tree(ns) -> None:
    root = FakeNode("root", reached_limit=True, stage="root")
    low = root.add_child(FakeNode("low", metric=0.4, reached_limit=False))
    high = root.add_child(FakeNode("high", metric=0.9, reached_limit=False))
    exhausted = root.add_child(FakeNode("exhausted", metric=0.99, reached_limit=True))
    agent = FakeAgent("greedy_tree", root, [low, high, exhausted])

    selected = ns.select_with_soft_switch(agent)

    assert_selected("greedy_tree", selected, high)
    assert agent.last_selection_rationale["reason"] == "best_expandable_valid_node"


def test_vanilla_mcts(ns) -> None:
    root = FakeNode("root", reached_limit=True, stage="root")
    weak = root.add_child(FakeNode("weak", reached_limit=False, uct=0.1, stage="draft"))
    strong = root.add_child(FakeNode("strong", reached_limit=False, uct=2.0, stage="draft"))
    agent = FakeAgent("vanilla_mcts", root, [weak, strong])

    selected = ns.select_with_soft_switch(agent)

    assert_selected("vanilla_mcts", selected, strong)
    assert agent.last_selection_rationale["policy"] == "vanilla_mcts"
    assert agent.last_selection_rationale["reason"] == "fixed_uct_from_root"


def test_progressive_mcts(ns) -> None:
    root = FakeNode("root", reached_limit=True, stage="root")
    weak = root.add_child(FakeNode("weak", reached_limit=False, uct=0.1, stage="draft"))
    strong = root.add_child(FakeNode("strong", reached_limit=False, uct=2.0, stage="draft"))
    agent = FakeAgent("progressive_mcts", root, [weak, strong])

    selected = ns.select_with_soft_switch(agent)

    assert_selected("progressive_mcts", selected, strong)
    assert agent.last_selection_rationale["policy"] == "progressive_mcts"
    assert agent.last_selection_rationale["reason"] == "decayed_uct_from_root"


def test_vanilla_mcts_exhausted_root_returns_sentinel(ns) -> None:
    root = FakeNode("root", reached_limit=True, stage="root")
    locked = root.add_child(FakeNode("locked", reached_limit=True, uct=2.0, stage="draft"))
    locked.lock = True
    agent = FakeAgent("vanilla_mcts", root, [locked])

    selected = ns.select_with_soft_switch(agent)

    assert_selected("vanilla_mcts_exhausted_root", selected, root)
    assert agent.last_selection_rationale["policy"] == "vanilla_mcts"


def test_default_mcgs_sets_rationale(ns) -> None:
    root = FakeNode("root", reached_limit=False, stage="root")
    agent = FakeAgent("mcgs", root, [])

    selected = ns.select_with_soft_switch(agent)

    assert_selected("mcgs", selected, root)
    assert agent.last_selection_rationale["policy"] == "mcgs"
    assert agent.last_selection_rationale["reason"] == "search_not_started_uct"


def test_novelty_lambda_changes_greedy_score(ns) -> None:
    root = FakeNode("root", reached_limit=True, stage="root")
    previous = root.add_child(FakeNode("previous", metric=0.7, reached_limit=True))
    previous.plan = "Use xgboost with tabular feature engineering"
    repeat = root.add_child(FakeNode("repeat", metric=0.9, reached_limit=False))
    repeat.plan = "Use xgboost with tabular feature engineering"
    distinct = root.add_child(FakeNode("distinct", metric=0.8, reached_limit=False))
    distinct.plan = "Use transformer image augmentation ensemble"

    baseline_agent = FakeAgent("greedy_tree", root, [previous, repeat, distinct], novelty_lambda=0.0)
    baseline_selected = ns.select_with_soft_switch(baseline_agent)
    assert_selected("novelty_lambda_0", baseline_selected, repeat)

    novelty_agent = FakeAgent("greedy_tree", root, [previous, repeat, distinct], novelty_lambda=0.4)
    novelty_selected = ns.select_with_soft_switch(novelty_agent)
    assert_selected("novelty_lambda_positive", novelty_selected, distinct)


def main() -> int:
    ns = load_node_selection()
    test_linear_chain(ns)
    test_greedy_tree(ns)
    test_vanilla_mcts(ns)
    test_progressive_mcts(ns)
    test_vanilla_mcts_exhausted_root_returns_sentinel(ns)
    test_default_mcgs_sets_rationale(ns)
    test_novelty_lambda_changes_greedy_score(ns)
    print("node selection mode tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
