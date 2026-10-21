"""Counterfactual supervision and offline label generation for Rex training (Step A9).

This module provides pure mathematical functions for offline trajectory label generation:
1. generate_sufficiency_label: Label S* in {0, 1} for epistemic sufficiency head.
2. generate_tool_gain_labels: Counterfactual information gain labels gain*_k in [0, 1] per tool.
3. generate_stall_label: Early learned STALL supervision indicator.
4. generate_redundancy_mask: Boolean mask filtering out redundant and refuted tool branches.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set

import torch

DEFAULT_S_STAR_THRESHOLD: float = 0.85


def generate_sufficiency_label(
    alive_hypotheses: Sequence[str],
    target_hypothesis: str,
    target_falsified: bool = False,
) -> float:
    """Compute binary ground-truth sufficiency label S* in {0.0, 1.0}.

    Condition:
        S* = 1.0 if the evidence is sufficient to conclusively determine the verdict:
             (1) Exactly one hypothesis survives and it is the target_hypothesis.
             (2) Or all cataloged hypotheses are eliminated and target is H_novel.
        Otherwise S* = 0.0.

    Args:
        alive_hypotheses: List or set of currently surviving hypothesis IDs.
        target_hypothesis: Ground truth winner hypothesis ID.
        target_falsified: Whether the true target was erroneously falsified.

    Returns:
        Float label 1.0 if sufficient, 0.0 otherwise.
    """
    if target_falsified:
        return 0.0

    alive_set = set(alive_hypotheses)
    if target_hypothesis not in alive_set:
        return 0.0

    # Singleton survivor
    if len(alive_set) == 1 and target_hypothesis in alive_set:
        return 1.0

    # OOD / Novel case: only H_novel remains among survivors
    known_alive = [h for h in alive_set if h != "H_novel"]
    if len(known_alive) == 0 and target_hypothesis == "H_novel":
        return 1.0

    return 0.0


def generate_tool_gain_labels(
    alive_hypotheses: Sequence[str],
    tool_specs: Sequence[ToolSpec],
    counterfactual_observations: Dict[str, Any],
    falsification_rules: Dict[str, List[FalsificationRule]],
) -> Dict[str, float]:
    """Compute counterfactual information gain gain*_k in [0.0, 1.0] for every tool.

    Formula:
        gain*_k = |{H in Alive : Falsified by Tool k}| / max(1, |Alive|)

    Args:
        alive_hypotheses: Currently surviving hypothesis IDs.
        tool_specs: Specifications of all candidate tools.
        counterfactual_observations: Complete unmasked ground-truth metric values from world rollout.
        falsification_rules: Mapping from hypothesis ID to its Popperian falsification rules.

    Returns:
        Dictionary mapping tool name to its counterfactual information gain in [0.0, 1.0].
    """
    alive_set = set(alive_hypotheses)
    total_alive = len(alive_set)
    if total_alive <= 1:
        # If 0 or 1 hypothesis survives, no further epistemic reduction is achievable
        return {spec.name: 0.0 for spec in tool_specs}

    gains: Dict[str, float] = {}

    for spec in tool_specs:
        tool_name = spec.name
        metrics_provided = set(spec.returns)

        # Check which currently alive hypotheses would be refuted by metrics provided by this tool
        eliminated_count = 0
        for hyp_id in alive_set:
            if hyp_id == "H_novel":
                # H_novel cannot be eliminated by standard falsification rules
                continue

            rules = falsification_rules.get(hyp_id, [])
            # If any rule belonging to this tool's returned metrics evaluates to True, hyp is eliminated
            hyp_refuted = False
            for rule in rules:
                if rule.metric in metrics_provided and rule.metric in counterfactual_observations:
                    obs_val = counterfactual_observations[rule.metric]
                    if rule.eval(obs_val):
                        hyp_refuted = True
                        break

            if hyp_refuted:
                eliminated_count += 1

        # Epistemic reduction ratio
        gains[tool_name] = float(eliminated_count) / float(total_alive)

    return gains


def generate_stall_label(
    s_star: float,
    tool_gains: Dict[str, float],
    tool_costs: Optional[Dict[str, float]] = None,
    s_threshold: float = DEFAULT_S_STAR_THRESHOLD,
) -> float:
    """Compute ground-truth learned STALL label in {0.0, 1.0}.

    Condition:
        STALL* = 1.0 iff S* < s_threshold AND all tools have gain*_k < tau_k.
        Otherwise STALL* = 0.0.

    Args:
        s_star: Ground truth sufficiency label (0.0 or 1.0).
        tool_gains: Counterfactual information gains per tool.
        tool_costs: Tool cost thresholds tau_k (default 0.0 per tool).
        s_threshold: Sufficiency cutoff threshold.

    Returns:
        1.0 if investigation is stagnated/unsolvable, 0.0 otherwise.
    """
    if s_star >= s_threshold:
        return 0.0

    costs = tool_costs or {}
    # Check if any tool produces sufficient gain above its cost hurdle
    for tool_name, gain in tool_gains.items():
        tau_k = costs.get(tool_name, 0.0)
        if gain > tau_k:
            return 0.0

    return 1.0


def generate_redundancy_mask(
    all_tools: Sequence[str],
    called_tools: Sequence[str],
    alive_hypotheses: Sequence[str],
    tool_hyp_mapping: Dict[str, List[str]],
) -> List[bool]:
    """Generate boolean redundancy mask of length N_tools.

    Value:
        True: Tool is redundant (already called or all its associated hypotheses are eliminated).
        False: Tool remains viable and uncalled.

    Args:
        all_tools: List of all tool names in catalog order.
        called_tools: List of tool names called in prior rounds.
        alive_hypotheses: Currently surviving hypothesis IDs.
        tool_hyp_mapping: Mapping from tool name to associated hypothesis IDs.

    Returns:
        List of booleans of length N_tools.
    """
    called_set = set(called_tools)
    alive_set = set(alive_hypotheses)
    redundancy: List[bool] = []

    for tool in all_tools:
        if tool in called_set:
            redundancy.append(True)
            continue

        associated_hyps = tool_hyp_mapping.get(tool, [])
        # If tool is associated with hypotheses, check if at least one is alive
        if associated_hyps:
            has_alive_hyp = any(h in alive_set for h in associated_hyps)
            redundancy.append(not has_alive_hyp)
        else:
            # General / unassociated tool is not redundant if uncalled
            redundancy.append(False)

    return redundancy
