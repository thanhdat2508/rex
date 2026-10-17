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

    LT = "LT"    # actual < threshold -> violation (eliminate)
    GT = "GT"    # actual > threshold -> violation (eliminate)
    EQ = "EQ"    # actual == threshold -> violation (eliminate)
    NEQ = "NEQ"  # actual != threshold -> violation (eliminate)


# Strategy registry mapping operators to evaluation callables (G23)
OPERATOR_STRATEGIES: Dict[ComparisonOperator, Callable[[float, float], bool]] = {
    ComparisonOperator.LT: lambda actual, threshold: actual < threshold,
    ComparisonOperator.GT: lambda actual, threshold: actual > threshold,
    ComparisonOperator.EQ: lambda actual, threshold: abs(actual - threshold) < 1e-6,
    ComparisonOperator.NEQ: lambda actual, threshold: abs(actual - threshold) >= 1e-6,
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
    threshold: float


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
            observations: Dict of metric names to measured scalar values.
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
                    actual_val = float(observations[rule.metric])
                    evaluator = OPERATOR_STRATEGIES.get(rule.operator)
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
    ) -> List[bool]:
        """Determine boolean availability for each tool based on surviving hypotheses.

        Rule: A tool is disabled (False) if and only if ALL hypotheses that it supports
        have been eliminated. If a tool supports multiple hypotheses (shared tool), it
        remains active as long as at least one supporting hypothesis is still alive.
        If a tool has no explicit mapping, it defaults to active (True).

        Args:
            tools: Ordered list of tool names known to the model.
            current_alive: Current boolean survival map for hypotheses.
            tool_hyp_mapping: Mapping of tool_name -> list of supported hypothesis IDs.

        Returns:
            List of boolean flags matching the order of tools.
        """
        tool_mask = []
        for tool_name in tools:
            supported_hyps = tool_hyp_mapping.get(tool_name, [])
            if not supported_hyps:
                # Unmapped tools remain active by default
                tool_mask.append(True)
                continue

            # Tool remains active if ANY of its supporting hypotheses is alive
            is_any_hyp_alive = any(current_alive.get(hyp, False) for hyp in supported_hyps)
            tool_mask.append(is_any_hyp_alive)

        return tool_mask
