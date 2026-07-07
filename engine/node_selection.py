"""Node selection: UCT select, get_exploration_weight, get_top_k_nodes_global, select_from_top_k_weighted, select_with_soft_switch."""

import logging
import random
import time
from typing import Any, List

from engine.search_node import SearchNode
from engine.conditions import should_trigger_branch_fusion
from agents.memory.ablation_controls import reset_memory_events
from agents.workflow_controls import operator_allowed
from utils.diversity_novelty import nearest_neighbor_distance, node_summary_text, previous_node_texts
logger = logging.getLogger("MLEvolve")


def _search_policy(agent) -> str:
    ablation_cfg = getattr(getattr(agent, "cfg", None), "ablation", None)
    return str(getattr(ablation_cfg, "search_policy", "mcgs") or "mcgs")


def _set_selection_rationale(agent, node: SearchNode, policy: str, reason: str, **extra: Any) -> SearchNode:
    reset_memory_events(agent)
    rationale = {
        "policy": policy,
        "selected_node_id": getattr(node, "id", None),
        "reason": reason,
        **extra,
    }
    setattr(agent, "last_selection_rationale", rationale)
    setattr(node, "_selection_rationale", rationale)
    logger.info(f"[select] -> node {node.id} (policy={policy}, reason={reason})")
    return node


def _metric_value(node: SearchNode) -> float | None:
    metric = getattr(node, "metric", None)
    return getattr(metric, "value", None) if metric is not None else None


def _score_for_sort(agent, node: SearchNode) -> float:
    value = _metric_value(node)
    if value is None:
        return float("-inf")
    score = float(value) if getattr(agent, "metric_maximize", True) else -float(value)
    novelty_lambda = _novelty_lambda(agent)
    if novelty_lambda > 0:
        score += novelty_lambda * _node_novelty(agent, node)
    return score


def _novelty_lambda(agent) -> float:
    ablation_cfg = getattr(getattr(agent, "cfg", None), "ablation", None)
    try:
        return float(getattr(ablation_cfg, "novelty_lambda", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _node_novelty(agent, node: SearchNode) -> float:
    cached = getattr(node, "novelty_score", None)
    if cached is not None:
        return float(cached)
    previous_text = previous_node_texts(_iter_known_nodes(agent), current_node=node)
    distance = nearest_neighbor_distance(node_summary_text(node), previous_text)
    novelty = 0.0 if distance is None else float(distance)
    setattr(node, "novelty_score", novelty)
    return novelty


def _stable_node_key(node: SearchNode) -> tuple:
    return (getattr(node, "step", 0) or 0, getattr(node, "ctime", 0.0) or 0.0, getattr(node, "id", ""))


def _best_by_metric(agent, nodes: list[SearchNode]) -> SearchNode | None:
    if not nodes:
        return None
    return max(nodes, key=lambda node: (_score_for_sort(agent, node), _stable_node_key(node)))


def _iter_known_nodes(agent) -> list[SearchNode]:
    seen = set()
    nodes: list[SearchNode] = []

    def add(node: SearchNode | None) -> None:
        if node is None or id(node) in seen:
            return
        seen.add(id(node))
        nodes.append(node)

    add(getattr(agent, "virtual_root", None))
    for branch_nodes in getattr(agent, "branch_all_nodes", {}).values():
        for node in branch_nodes:
            add(node)
    for node in getattr(getattr(agent, "journal", None), "nodes", []):
        add(node)
    return nodes


def select_linear_chain(agent) -> SearchNode:
    """Select along one best path, expanding the leaf before creating alternatives."""
    node = agent.virtual_root
    path = [getattr(node, "id", None)]

    while node is not None and not getattr(node, "is_terminal", False):
        if not node.reached_child_limit(scfg=agent.scfg):
            return _set_selection_rationale(
                agent,
                node,
                "linear_chain",
                "expand_first_available_node_on_best_path",
                path=path,
            )
        child = _best_by_metric(agent, list(getattr(node, "children", [])))
        if child is None:
            return _set_selection_rationale(
                agent,
                node,
                "linear_chain",
                "no_child_available_after_limit",
                path=path,
            )
        node = child
        path.append(getattr(node, "id", None))

    return _set_selection_rationale(agent, agent.virtual_root, "linear_chain", "fallback_to_root", path=path)


def select_greedy_tree(agent) -> SearchNode:
    """Select the best currently expandable valid node without UCT exploration."""
    candidates = []
    for node in _iter_known_nodes(agent):
        if getattr(node, "is_terminal", False):
            continue
        if node.reached_child_limit(scfg=agent.scfg):
            continue
        if node is agent.virtual_root:
            candidates.append(node)
            continue
        if getattr(node, "is_buggy", None) is False and _metric_value(node) is not None:
            candidates.append(node)

    selected = _best_by_metric(agent, candidates)
    if selected is None:
        selected = agent.virtual_root
        reason = "no_expandable_valid_node"
    elif selected is agent.virtual_root:
        reason = "root_has_remaining_draft_capacity"
    else:
        reason = "best_expandable_valid_node"
    return _set_selection_rationale(
        agent,
        selected,
        "greedy_tree",
        reason,
        candidate_count=len(candidates),
        metric_value=_metric_value(selected),
    )


def select_vanilla_mcts(agent) -> SearchNode:
    """Select with fixed-constant UCT from the root; configs disable MCGS extras."""
    previous_c = getattr(agent, "_fixed_exploration_constant", None)
    agent._fixed_exploration_constant = 1.414
    try:
        selected = select(agent, agent.virtual_root)
    finally:
        if previous_c is None:
            delattr(agent, "_fixed_exploration_constant")
        else:
            agent._fixed_exploration_constant = previous_c
    return _set_selection_rationale(agent, selected, "vanilla_mcts", "fixed_uct_from_root")


def select_progressive_mcts(agent) -> SearchNode:
    """Select with decayed UCT from the root without MCGS top-K switching."""
    selected = select(agent, agent.virtual_root)
    return _set_selection_rationale(agent, selected, "progressive_mcts", "decayed_uct_from_root")


def _piecewise_decay(t, initial_C=1.414, T1=100, T2=200, alpha=0.01, lower_bound=0.7):
    """Piecewise decay: initial_C until T1, linear to lower_bound by T2, then lower_bound."""
    if t < T1:
        return initial_C
    elif T1 <= t <= T2:
        return max(initial_C - alpha * (t - T1), lower_bound)
    else:
        return lower_bound


def _compute_exploration_constant(agent):
    """Compute exploration constant C from search progress (piecewise decay)."""
    fixed_c = getattr(agent, "_fixed_exploration_constant", None)
    if fixed_c is not None:
        return fixed_c
    dcfg = agent.cfg.agent.decay
    n1 = agent.scfg.num_drafts * (agent.scfg.num_improves ** 2)
    n2 = round(agent.acfg.steps * dcfg.phase_ratios[0])
    t1 = min(n1, n2)
    t2 = round(agent.acfg.steps * dcfg.phase_ratios[1])
    return _piecewise_decay(
        t=agent.current_step,
        initial_C=dcfg.exploration_constant,
        T1=t1,
        T2=t2,
        alpha=dcfg.alpha,
        lower_bound=dcfg.lower_bound,
    )


def select(agent, node: SearchNode):
    """UCT selection: recurse from node, return node to expand (root lock for drafts)."""
    def _best_child(n: SearchNode) -> SearchNode:
        C = _compute_exploration_constant(agent)
        if agent.is_root(n):
            filtered_children = [child for child in n.children if not child.lock]
            selected_node = n
            if len(filtered_children) > 0:
                selected_node = max(filtered_children,
                                    key=lambda child: child.uct_value(exploration_constant=C))
            if selected_node.stage in ["draft", "fusion_draft"]:
                selected_node.lock = True
            return selected_node
        else:
            return max(n.children, key=lambda child: child.uct_value(exploration_constant=C))

    while node and not node.is_terminal:
        if not node.reached_child_limit(scfg=agent.scfg):
            if node.is_buggy and node.is_debug_success is True:
                node = _best_child(node)
            elif node.continue_improve and len(node.children) > 0:
                node = _best_child(node)
            else:
                logger.info(f"[select] → node {node.id} (method=expand)")
                return node
        else:
            if (
                agent.is_root(node)
                and getattr(agent.acfg, "use_aggregation", True)
                and operator_allowed(agent, "Aggregation")
                and should_trigger_branch_fusion(agent)
                and random.random() < agent.acfg.branch_fusion_trigger_prob
            ):
                logger.info(f"Root node {node.id} is fully expanded for regular drafts, aggregation conditions met (including probability), returning root")
                return node
            node = _best_child(node)
    logger.info(f"[select] → node {node.id} (method=uct)")
    return node


def get_exploration_weight(time_elapsed: float, total_time: float,
                           switch_start: float = 0.5,
                           switch_end: float = 0.7,
                           min_weight: float = 0.2) -> float:
    """Exploration weight: 1.0 until switch_start, linear decay to min_weight by switch_end."""
    time_progress = time_elapsed / total_time

    if time_progress < switch_start:
        return 1.0
    elif time_progress < switch_end:
        decay_progress = (time_progress - switch_start) / (switch_end - switch_start)
        return 1.0 - (1.0 - min_weight) * decay_progress
    else:
        return min_weight


def get_top_k_nodes_global(agent, k: int, max_from_same_branch: int) -> List[dict]:
    """Select top-k nodes globally with branch diversity (recomputed each call). Returns list of {node, branch_id, metric, rank}."""
    all_nodes = []
    for branch_id in agent.branch_all_nodes:
        for node in agent.branch_all_nodes[branch_id]:
            if not node.is_buggy and node.metric is not None and node.metric.value is not None:
                all_nodes.append(node)

    if not all_nodes:
        logger.warning("No valid nodes found for Top-K selection")
        return []

    maximize = agent.metric_maximize
    if _novelty_lambda(agent) > 0:
        all_nodes.sort(key=lambda n: _score_for_sort(agent, n), reverse=True)
    else:
        all_nodes.sort(
            key=lambda n: n.metric.value,
            reverse=maximize
        )

    logger.info(f"Total valid nodes: {len(all_nodes)}, requesting Top-{k}")

    selected = []
    branch_count = {}

    for node in all_nodes:
        if len(selected) >= k:
            break

        branch_id = node.branch_id
        current_count = branch_count.get(branch_id, 0)

        if current_count >= max_from_same_branch:
            logger.debug(f"Branch {branch_id} reached limit ({max_from_same_branch}), skipping node with metric={node.metric.value:.4f}")
            continue

        selected.append({
            'node': node,
            'branch_id': branch_id,
            'metric': node.metric.value,
            'rank': len(selected) + 1
        })
        branch_count[branch_id] = current_count + 1

    if selected:
        branch_distribution = {}
        for item in selected:
            bid = item['branch_id']
            branch_distribution[bid] = branch_distribution.get(bid, 0) + 1

        metrics_str = ", ".join([f"Rank{item['rank']}={item['metric']:.4f}(B{item['branch_id']})" for item in selected])
        logger.info(f"📊 Top-{len(selected)} selected: {metrics_str}")
        logger.info(f"📊 Branch distribution: {branch_distribution}")

    return selected


def select_from_top_k_weighted(agent, top_k_nodes: List[dict]) -> SearchNode:
    """Weighted random choice from top-k nodes (weight = 1/rank)."""
    if not top_k_nodes:
        return select(agent, agent.virtual_root)

    weights = [1.0 / item['rank'] for item in top_k_nodes]
    total_weight = sum(weights)
    probabilities = [w / total_weight for w in weights]
    selected = random.choices(top_k_nodes, weights=probabilities)[0]

    logger.info(f"🎯 Selected: Rank{selected['rank']} (Branch {selected['branch_id']}, "
                f"metric={selected['metric']:.4f}, prob={probabilities[top_k_nodes.index(selected)]:.1%})")

    return selected['node']


def select_with_soft_switch(agent) -> SearchNode:
    """Soft switch: exploration (UCT) vs exploitation (Top-K) by time progress."""
    policy = _search_policy(agent)
    if policy == "linear_chain":
        return select_linear_chain(agent)
    if policy == "greedy_tree":
        return select_greedy_tree(agent)
    if policy == "vanilla_mcts":
        return select_vanilla_mcts(agent)
    if policy == "progressive_mcts":
        return select_progressive_mcts(agent)
    if policy == "single_shot":
        return _set_selection_rationale(agent, agent.virtual_root, "single_shot", "draft_only_root_selection")
    if policy != "mcgs":
        logger.warning(f"Unknown search_policy={policy!r}; falling back to mcgs")

    if agent.search_start_time is None:
        logger.info("📊 Search not started yet, using standard UCT")
        selected = select(agent, agent.virtual_root)
        return _set_selection_rationale(agent, selected, "mcgs", "search_not_started_uct")

    time_elapsed = time.time() - agent.search_start_time
    total_time = agent.acfg.time_limit
    time_progress = time_elapsed / total_time

    scfg = agent.scfg

    exploration_weight = get_exploration_weight(
        time_elapsed, total_time,
        switch_start=scfg.explore_switch_start,
        switch_end=scfg.explore_switch_end,
        min_weight=scfg.min_exploration_weight,
    )

    if random.random() < exploration_weight:
        logger.info(f"📊 Exploration mode (weight={exploration_weight:.2%}, "
                   f"time={time_progress:.1%})")
        selected = select(agent, agent.virtual_root)
        return _set_selection_rationale(
            agent,
            selected,
            "mcgs",
            "exploration_uct",
            exploration_weight=exploration_weight,
            time_progress=time_progress,
        )

    else:
        # Top-K exploitation
        logger.info(f"🎯 Exploitation mode (weight={1-exploration_weight:.2%}, "
                   f"time={time_progress:.1%})")

        if time_progress < scfg.explore_switch_end:
            k = scfg.topk_early_k
            max_from_same_branch = scfg.topk_early_max_per_branch
            phase = f"early-mid (<{scfg.explore_switch_end:.0%})"
        else:
            k = scfg.topk_late_k
            max_from_same_branch = scfg.topk_late_max_per_branch
            phase = f"late (>={scfg.explore_switch_end:.0%})"

        logger.info(f"📊 Phase: {phase}, requesting Top-{k} (max {max_from_same_branch} per branch)")

        top_k_nodes = get_top_k_nodes_global(
            agent,
            k=k,
            max_from_same_branch=max_from_same_branch
        )

        if not top_k_nodes:
            logger.warning("No valid Top-K nodes found, fallback to standard UCT")
            selected = select(agent, agent.virtual_root)
            return _set_selection_rationale(
                agent,
                selected,
                "mcgs",
                "topk_empty_fallback_uct",
                exploration_weight=exploration_weight,
                time_progress=time_progress,
            )

        available_nodes = [
            item for item in top_k_nodes
            if not item['node'].reached_child_limit(agent.scfg, for_topk=True)
        ]

        if available_nodes:
            selected_node = select_from_top_k_weighted(agent, available_nodes)
            logger.info(f"✅ Selected unexpanded Top-K node {selected_node.id} (from {len(available_nodes)}/{len(top_k_nodes)} available)")
            selected_node._topk_triggered = True
            return _set_selection_rationale(
                agent,
                selected_node,
                "mcgs",
                "topk_weighted_unexpanded",
                exploration_weight=exploration_weight,
                time_progress=time_progress,
                topk_size=len(top_k_nodes),
                available_topk_size=len(available_nodes),
            )
        else:
            logger.info(f"⚠️ All Top-{len(top_k_nodes)} nodes fully expanded, will apply UCT from selected node")
            selected_node = select_from_top_k_weighted(agent, top_k_nodes)
            logger.info(f"Selected fully expanded node {selected_node.id}, applying UCT from it")
            uct_node = select(agent, selected_node)
            uct_node._topk_triggered = True
            return _set_selection_rationale(
                agent,
                uct_node,
                "mcgs",
                "topk_weighted_then_uct",
                exploration_weight=exploration_weight,
                time_progress=time_progress,
                topk_size=len(top_k_nodes),
            )
