"""Counterfactual supervision and offline label generation for Rex training (Step A9).

This module provides pure mathematical functions for offline trajectory label generation:
1. generate_sufficiency_label: Label S* in {0, 1} for epistemic sufficiency head.
2. generate_tool_gain_labels: Counterfactual information gain labels gain*_k in {0, 1, -100} per tool.
3. generate_stall_label: Early learned STALL supervision indicator.
4. generate_redundancy_mask: Boolean mask filtering out redundant and refuted tool branches.

Label semantics follow docs/rex-label-schema.md.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

if TYPE_CHECKING:
    from .rex_registry import FalsificationRule, ToolSpec

DEFAULT_S_STAR_THRESHOLD: float = 0.85
GAIN_UNOBSERVED: float = -100.0

# Default business risk per domain type token, used when the registry domain sets no `risk`.
DEFAULT_RISK_MAPPING: Dict[str, float] = {
    "<TYPE_PAYMENT_FRAUD>": 0.5,
    "<TYPE_ACCOUNT_TAKEOVER>": 0.4,
    "<TYPE_PROMO_ABUSE>": 0.1,
    "<TYPE_API_SCRAPING>": 0.05,
    "<TYPE_CREDENTIAL_STUFFING>": 0.1,
}


def generate_sufficiency_label(
    alive_hypotheses: Sequence[str],
    target_hypothesis: str,
    target_falsified: bool = False,
) -> float:
    """Compute binary ground-truth sufficiency label S* in {0.0, 1.0}.

    Condition:
        S* = 1.0 if the evidence is sufficient to conclusively determine the verdict:
             (1) The target is the only surviving cataloged hypothesis.
             (2) Or all cataloged hypotheses are eliminated and target is H_novel.
        Otherwise S* = 0.0.

        H_novel is never eliminated, so it is not counted as a competing survivor.

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

    known_alive = alive_set - {"H_novel"}
    if target_hypothesis == "H_novel":
        # OOD / Novel case: every cataloged hypothesis has been eliminated
        return 1.0 if not known_alive else 0.0

    # Cataloged target is the only cataloged survivor
    return 1.0 if known_alive == {target_hypothesis} else 0.0


def generate_tool_gain_labels(
    alive_hypotheses: Sequence[str],
    tool_specs: Sequence[ToolSpec],
    counterfactual_observations: Dict[str, Any],
    falsification_rules: Dict[str, List[FalsificationRule]],
    mask_unobserved: bool = False,
) -> Dict[str, float]:
    """Compute binary counterfactual information gain gain*_k in {0.0, 1.0, -100.0} for every tool.

    Definition (docs/rex-label-schema.md, labels.tool_gains):
        gain*_k = 1.0 if tool k's observations would falsify at least one surviving cataloged hypothesis
                  (it uncovers an unseen key fact), 0.0 otherwise.
        If mask_unobserved=True and none of tool k's metrics were recorded, gain*_k = -100.0 (masked from loss).

    Args:
        alive_hypotheses: Currently surviving hypothesis IDs.
        tool_specs: Specifications of all candidate tools.
        counterfactual_observations: Complete unmasked ground-truth metric values from world rollout.
        falsification_rules: Mapping from hypothesis ID to its Popperian falsification rules.
        mask_unobserved: Whether to assign -100.0 to tools whose metrics were unobserved.

    Returns:
        Dictionary mapping tool name to its counterfactual information gain (1.0, 0.0 or -100.0).
    """
    # H_novel cannot be eliminated by standard falsification rules
    known_alive = set(alive_hypotheses) - {"H_novel"}

    gains: Dict[str, float] = {}

    for spec in tool_specs:
        tool_name = spec.name
        metrics_provided = set(spec.returns)

        observed_metrics = [m for m in metrics_provided if m in counterfactual_observations]
        if mask_unobserved and not observed_metrics and metrics_provided:
            gains[tool_name] = GAIN_UNOBSERVED
            continue

        refutes_any = False
        for hyp_id in known_alive:
            for rule in falsification_rules.get(hyp_id, []):
                if rule.metric in metrics_provided and rule.metric in counterfactual_observations:
                    if rule.eval(counterfactual_observations[rule.metric]):
                        refutes_any = True
                        break
            if refutes_any:
                break

        gains[tool_name] = 1.0 if refutes_any else 0.0

    return gains


def generate_stall_label(
    s_star: float,
    tool_gains: Dict[str, float],
    tool_costs: Optional[Dict[str, float]] = None,
    s_threshold: float = DEFAULT_S_STAR_THRESHOLD,
    allowed_tools: Optional[Sequence[str]] = None,
) -> float:
    """Compute ground-truth learned STALL label in {0.0, 1.0}.

    Condition (docs/rex-label-schema.md, labels.is_stall):
        STALL* = 1.0 iff S* < s_threshold AND no remaining allowed tool has gain*_k > 0.
        Otherwise STALL* = 0.0.

    The cost hurdle tau_k is a dispatcher concern applied to the predicted gain probability
    (rex_tools.gain_threshold); the label only records whether a tool can still uncover a key fact.

    Args:
        s_star: Ground truth sufficiency label (0.0 or 1.0).
        tool_gains: Counterfactual information gains per tool.
        tool_costs: Unused; kept for call compatibility.
        s_threshold: Sufficiency cutoff threshold.
        allowed_tools: Tools still callable (uncalled and not PoE-masked). None means every tool.

    Returns:
        1.0 if investigation is stagnated/unsolvable, 0.0 otherwise.
    """
    if s_star >= s_threshold:
        return 0.0

    allowed = set(allowed_tools) if allowed_tools is not None else None
    for tool_name, gain in tool_gains.items():
        if gain == GAIN_UNOBSERVED:
            continue
        if allowed is not None and tool_name not in allowed:
            continue
        if gain > 0.0:
            return 0.0

    return 1.0


def generate_risk_label(
    type_token: str,
    risk_mapping: Optional[Dict[str, float]] = None,
    default_risk: float = 0.1,
) -> float:
    """Compute target risk label R* in [0.0, 1.0] based on domain business policy."""
    mapping = risk_mapping or DEFAULT_RISK_MAPPING
    return mapping.get(type_token, default_risk)


def generate_hypotheses_survival_labels(
    alive_hypotheses: Sequence[str],
    all_hypotheses: Sequence[str],
) -> List[float]:
    """Compute binary survival indicators hyp_true in {0.0, 1.0} across all candidate hypotheses."""
    alive_set = set(alive_hypotheses)
    return [1.0 if h in alive_set else 0.0 for h in all_hypotheses]


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
