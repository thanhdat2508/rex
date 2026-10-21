"""Karl Popper Process-of-Elimination (PoE) engine and data contracts for Rex.

This module implements:
1. ComparisonOperator and Strategy Pattern for deterministic falsification rules.
2. Data contracts: FalsificationRule, HypothesisDefinition, RoundTrace, InvestigationResult.
3. PoEFalsificationEngine: Evaluates telemetry against hypothesis falsification criteria.
4. HardMaskingManager: Maps hypothesis survival to tool availability masks.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import torch

# Comparison operators for deterministic falsification criteria (P2)
class ComparisonOperator(str, Enum):
    """Supported relational operators for hypothesis falsification rules."""

    LT = "LT"        # actual < threshold -> violation (eliminate)
    GT = "GT"        # actual > threshold -> violation (eliminate)
    GTE = "GTE"      # actual >= threshold -> violation (eliminate)
    LTE = "LTE"      # actual <= threshold -> violation (eliminate)
    EQ = "EQ"        # actual == threshold -> violation (eliminate)
    NEQ = "NEQ"      # actual != threshold -> violation (eliminate)
    IN = "IN"        # actual in threshold -> violation (eliminate)
    NOT_IN = "NOT_IN"# actual not in threshold -> violation (eliminate)


def _evaluate_operator(actual: Any, threshold: Any, op: str) -> bool:
    """Evaluate observation against threshold with support for numeric, boolean, and collection types."""
    if actual is None:
        return False
    if op == "GT":
        return float(actual) > float(threshold)
    elif op == "LT":
        return float(actual) < float(threshold)
    elif op == "GTE":
        return float(actual) >= float(threshold)
    elif op == "LTE":
        return float(actual) <= float(threshold)
    elif op == "EQ":
        if isinstance(threshold, bool) or isinstance(actual, bool):
            return bool(actual) is bool(threshold)
        return actual == threshold
    elif op == "NEQ":
        if isinstance(threshold, bool) or isinstance(actual, bool):
            return bool(actual) is not bool(threshold)
        return actual != threshold
    elif op == "IN":
        return actual in threshold
    elif op == "NOT_IN":
        return actual not in threshold
    return False


# Strategy registry mapping operators to evaluation callables (G23)
OPERATOR_STRATEGIES: Dict[ComparisonOperator, Callable[[Any, Any], bool]] = {
    ComparisonOperator.LT: lambda actual, threshold: _evaluate_operator(actual, threshold, "LT"),
    ComparisonOperator.GT: lambda actual, threshold: _evaluate_operator(actual, threshold, "GT"),
    ComparisonOperator.GTE: lambda actual, threshold: _evaluate_operator(actual, threshold, "GTE"),
    ComparisonOperator.LTE: lambda actual, threshold: _evaluate_operator(actual, threshold, "LTE"),
    ComparisonOperator.EQ: lambda actual, threshold: _evaluate_operator(actual, threshold, "EQ"),
    ComparisonOperator.NEQ: lambda actual, threshold: _evaluate_operator(actual, threshold, "NEQ"),
    ComparisonOperator.IN: lambda actual, threshold: _evaluate_operator(actual, threshold, "IN"),
    ComparisonOperator.NOT_IN: lambda actual, threshold: _evaluate_operator(actual, threshold, "NOT_IN"),
}


class TerminationReason(str, Enum):
    """Reasons for terminating an iterative investigation loop (P2)."""

    SINGLE_SURVIVOR = "single_survivor"
    MAX_ROUNDS_REACHED = "max_rounds_reached"
    EMPTY_TOOL_SELECTION = "empty_tool_selection"
    ALL_CATALOGED_ELIMINATED = "all_cataloged_eliminated"


@dataclass(frozen=True)
class FalsificationRule:
    """Individual Popperian falsification criterion (F1).

    If an observation metric violates this rule, the associated hypothesis
    is eliminated (falsified).
    """

    metric: str
    operator: ComparisonOperator
    threshold: Any


@dataclass(frozen=True)
class HypothesisDefinition:
    """Domain hypothesis specification for MECE ontology registry (F1)."""

    id: str
    text: str
    associated_tools: List[str]
    falsification_rules: List[FalsificationRule] = field(default_factory=list)
    is_residual: bool = False


@dataclass(frozen=True)
class RoundTrace:
    """Telemetry and decision trace recorded during one investigation round (F1)."""

    round_idx: int
    selected_tools: List[str]
    observations: Dict[str, Any]
    active_hypotheses: List[str]


@dataclass(frozen=True)
class InvestigationResult:
    """Final decision output produced by Rex autonomous investigation loop (F1)."""

    predicted_hypothesis: str
    probabilities: Dict[str, float]
    rounds_executed: int
    termination_reason: TerminationReason
    is_escalated: bool
    traces: List[RoundTrace] = field(default_factory=list)


class PoEFalsificationEngine:
    """Karl Popper Falsification Engine.

    Evaluates telemetry observations against candidate hypothesis criteria.
    Hypotheses are never proven true; they can only survive falsification.
    H_novel is the open-world residual slot and is immune to static elimination.
    """

    def evaluate(
        self,
        observations: Dict[str, Any],
        current_alive: Dict[str, bool],
        rules: Dict[str, List[FalsificationRule]],
    ) -> Dict[str, bool]:
        """Evaluate observation telemetry and return updated survival flags.

        Args:
            observations: Dict of metric names to measured scalar/categorical values.
            current_alive: Current boolean survival state per hypothesis.
            rules: Mapping of hypothesis ID to its list of falsification rules.

        Returns:
            New dictionary with updated survival flags (pure function, no mutation).
        """
        updated_alive = dict(current_alive)

        for hyp_id, is_alive in current_alive.items():
            # Already eliminated hypotheses remain eliminated (idempotent)
            if not is_alive:
                continue

            # Residual hypothesis (H_novel) cannot be eliminated by static rules
            if hyp_id == "H_novel":
                updated_alive["H_novel"] = True
                continue

            hyp_rules = rules.get(hyp_id, [])
            for rule in hyp_rules:
                if rule.metric in observations:
                    actual_val = observations[rule.metric]
                    # Convert rule operator if it's a string
                    op_key = (
                        rule.operator
                        if isinstance(rule.operator, ComparisonOperator)
                        else ComparisonOperator(str(rule.operator))
                    )
                    evaluator = OPERATOR_STRATEGIES.get(op_key)
                    if evaluator is None:
                        raise ValueError(f"Unsupported operator '{rule.operator}' for metric '{rule.metric}'")

                    # If the measured telemetry satisfies the elimination condition, eliminate
                    if evaluator(actual_val, rule.threshold):
                        updated_alive[hyp_id] = False
                        break

        # Invariant: H_novel must always remain alive
        updated_alive["H_novel"] = True
        return updated_alive


class HardMaskingManager:
    """Derives tool and verdict availability masks from PoE survival state."""

    def compute_tool_mask(
        self,
        tools: List[str],
        current_alive: Dict[str, bool],
        tool_hyp_mapping: Dict[str, List[str]],
        called_tools: Optional[Any] = None,
    ) -> List[bool]:
        """Determine boolean availability for each tool based on surviving hypotheses.

        Rule: A tool is disabled (False) if:
        1. It has already been called in prior turns (non-repetition).
        2. ALL hypotheses that it supports have been eliminated.
        If a tool supports multiple hypotheses (shared tool), it remains active as long
        as at least one supporting hypothesis is still alive.
        If a tool has no explicit mapping, it defaults to active (True) unless called.

        Args:
            tools: Ordered list of tool names known to the model.
            current_alive: Current boolean survival map for hypotheses.
            tool_hyp_mapping: Mapping of tool_name -> list of supported hypothesis IDs.
            called_tools: Optional sequence or set of tool names executed in prior rounds.

        Returns:
            List of boolean flags matching the order of tools.
        """
        called_set = set(called_tools or [])
        tool_mask = []
        for tool_name in tools:
            if tool_name in called_set:
                tool_mask.append(False)
                continue

            supported_hyps = tool_hyp_mapping.get(tool_name, [])
            if not supported_hyps:
                # Unmapped tools remain active by default
                tool_mask.append(True)
                continue

            # Tool remains active if ANY of its supporting hypotheses is alive
            is_any_hyp_alive = any(current_alive.get(hyp, False) for hyp in supported_hyps)
            tool_mask.append(is_any_hyp_alive)

        return tool_mask


def apply_poe_logit_mask(
    logits: torch.Tensor,
    hyp_alive: torch.Tensor,
    masked_val: float = -1e4,
) -> torch.Tensor:
    """Apply hard Popperian logit mask to eliminated hypotheses.

    Formula:
        Logits_final = Logits_raw + (M_elim - 1) * 1e4
    """
    mask = hyp_alive.to(dtype=logits.dtype, device=logits.device)
    return logits + (mask - 1.0) * (-masked_val)


class HybridPoEEngine:
    """Hybrid Process-of-Elimination combining symbolic rules with neural hypothesis scoring (Step A6)."""

    def __init__(self, default_theta_elim: float = 0.20):
        self.default_theta_elim = default_theta_elim
        self.falsification_engine = PoEFalsificationEngine()

    def evaluate(
        self,
        neural_scores: Optional[torch.Tensor],
        hyp_keys: List[str],
        observations: Dict[str, Any],
        rules: Dict[str, List[FalsificationRule]],
        theta_elim: Optional[float] = None,
    ) -> Dict[str, bool]:
        """Compute combined survival map across candidate hypotheses.

        Properties:
            - Symbolic rules maintain absolute veto power if criteria are breached.
            - Hypotheses with neural score < theta_elim are eliminated.
            - H_novel is unconditionally preserved (open-world residual slot).
            - Supports hypothesis revival across rounds as full memory M grows.
        """
        thresh = theta_elim if theta_elim is not None else self.default_theta_elim
        current_alive = {k: True for k in hyp_keys}

        # 1. Symbolic rule falsification (hard veto)
        rule_alive = self.falsification_engine.evaluate(observations, current_alive, rules)

        # 2. Neural score gating
        hybrid_alive: Dict[str, bool] = {}
        if neural_scores is not None:
            scores_flat = neural_scores.squeeze(0).tolist() if neural_scores.dim() > 1 else neural_scores.tolist()
            for i, k in enumerate(hyp_keys):
                if k == "H_novel":
                    hybrid_alive["H_novel"] = True
                    continue
                score_survives = (scores_flat[i] >= thresh) if i < len(scores_flat) else True
                hybrid_alive[k] = rule_alive.get(k, True) and score_survives
        else:
            hybrid_alive = rule_alive

        # Invariant: H_novel must always remain alive
        hybrid_alive["H_novel"] = True
        return hybrid_alive
