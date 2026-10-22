"""Deterministic stop engine and escalation decision matrix for Rex Orchestration (B8).

Evaluates stopping conditions, risk barriers, budget limits, stagnation rules,
and produces standardized Decision outcomes for the runner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .agent_port import AgentCommand
from .guard import Guard


@dataclass(frozen=True)
class StopConfig:
    """Operational hyperparameters for stopping and escalation."""
    max_rounds: int = 3
    n_max_calls: int = 6
    risk_threshold: float = 0.30
    epsilon_s: float = 0.05
    # Off by default: the model's conformal auto-commit (|C| == 1, S < 0.85) is escalated until
    # q_hat and the verdict temperature are calibrated on real cases, not only on simulated rollouts.
    accept_conformal_verdict: bool = False


@dataclass(frozen=True)
class Decision:
    """Orchestration decision taken after evaluating an agent command."""
    action: str                                  # 'CONTINUE', 'AUTO_VERDICT', 'ESCALATE'
    tools: List[str] = field(default_factory=list)
    reason: Optional[str] = None                 # HIGH_RISK, DEADLOCK, BUDGET_EXHAUSTED, TOOL_BLOCKED
    detail: Optional[str] = None                 # CONFORMAL_LOW_S, NO_CALLABLE_TOOL, EPSILON_STAGNATION, etc.
    other_reasons: List[str] = field(default_factory=list)


def check_epsilon_stagnation(s_history: List[float], epsilon: float = 0.05) -> bool:
    """Check if sufficiency S increased by less than epsilon across two consecutive turns."""
    if len(s_history) < 3:
        return False

    # Check last two transitions: (s[-2] - s[-3]) < eps and (s[-1] - s[-2]) < eps
    delta1 = s_history[-2] - s_history[-3]
    delta2 = s_history[-1] - s_history[-2]
    return delta1 < epsilon and delta2 < epsilon


def decide(
    cmd: AgentCommand,
    s_history: List[float],
    n_calls: int,
    round_k: int,
    time_left_s: float,
    high_risk: bool,
    cfg: StopConfig,
    guard: Optional[Guard] = None,
    tools_cfg: Optional[Dict[str, Any]] = None,
    available_mask: Optional[Dict[str, bool]] = None,
    called_tools: Optional[List[str]] = None,
) -> Decision:
    """Pure, deterministic stop decision function without I/O."""
    other_reasons: List[str] = []

    # 1. Update HIGH_RISK status: rule floor tripped or model risk >= threshold
    active_high_risk = high_risk or (cmd.risk >= cfg.risk_threshold)

    # 2. Check S >= 0.85 conditions
    if cmd.s >= 0.85:
        if active_high_risk:
            # High risk requires human reviewer escalation even with sufficient evidence
            return Decision(
                action="ESCALATE",
                reason="HIGH_RISK",
                detail="HIGH_RISK_THRESHOLD_EXCEEDED",
                other_reasons=other_reasons,
            )
        if cmd.kind == "VERDICT":
            return Decision(
                action="AUTO_VERDICT",
                reason=None,
                detail=None,
                other_reasons=other_reasons,
            )

    # 3. Check VERDICT when S < 0.85
    if cmd.kind == "VERDICT":
        if not cfg.accept_conformal_verdict:
            return Decision(
                action="ESCALATE",
                reason="DEADLOCK",
                detail="CONFORMAL_LOW_S",
                other_reasons=other_reasons,
            )
        # Conformal verdict accepted
        if active_high_risk:
            return Decision(
                action="ESCALATE",
                reason="HIGH_RISK",
                detail="CONFORMAL_HIGH_RISK",
                other_reasons=other_reasons,
            )
        return Decision(
            action="AUTO_VERDICT",
            reason=None,
            detail="CONFORMAL_VERDICT_ACCEPTED",
            other_reasons=other_reasons,
        )

    # 4. Check STALL signal or epsilon stagnation
    if cmd.kind == "STALL":
        if active_high_risk:
            other_reasons.append("HIGH_RISK")
        return Decision(
            action="ESCALATE",
            reason="DEADLOCK",
            detail="STALL_SIGNAL",
            other_reasons=other_reasons,
        )

    if check_epsilon_stagnation(s_history, cfg.epsilon_s):
        if active_high_risk:
            other_reasons.append("HIGH_RISK")
        return Decision(
            action="ESCALATE",
            reason="DEADLOCK",
            detail="EPSILON_STAGNATION",
            other_reasons=other_reasons,
        )

    # 5. Check Budget limits: max_rounds, n_max_calls, or time budget
    if round_k >= cfg.max_rounds:
        if active_high_risk:
            other_reasons.append("HIGH_RISK")
        return Decision(
            action="ESCALATE",
            reason="BUDGET_EXHAUSTED",
            detail="MAX_ROUNDS_REACHED",
            other_reasons=other_reasons,
        )

    if n_calls >= cfg.n_max_calls:
        if active_high_risk:
            other_reasons.append("HIGH_RISK")
        return Decision(
            action="ESCALATE",
            reason="BUDGET_EXHAUSTED",
            detail="MAX_CALLS_REACHED",
            other_reasons=other_reasons,
        )

    if time_left_s <= 0:
        if active_high_risk:
            other_reasons.append("HIGH_RISK")
        return Decision(
            action="ESCALATE",
            reason="BUDGET_EXHAUSTED",
            detail="DEADLINE_EXCEEDED",
            other_reasons=other_reasons,
        )

    # 6. Action is CALL_TOOL: filter proposed tools through Guard
    if cmd.kind == "CALL_TOOL":
        active_guard = guard if guard is not None else Guard()
        if tools_cfg is not None:
            mask = available_mask or {}
            approved_tools, fail_reason, fail_detail = active_guard.filter_selected_tools(
                selected_tools=cmd.tools,
                tools_cfg=tools_cfg,
                available_mask=mask,
                called_tools=called_tools,
            )
            if not approved_tools:
                if active_high_risk:
                    other_reasons.append("HIGH_RISK")
                return Decision(
                    action="ESCALATE",
                    reason=fail_reason or "DEADLOCK",
                    detail=fail_detail or "NO_CALLABLE_TOOL",
                    other_reasons=other_reasons,
                )
            return Decision(
                action="CONTINUE",
                tools=approved_tools,
                reason=None,
                detail=None,
                other_reasons=other_reasons,
            )
        else:
            if not cmd.tools:
                return Decision(
                    action="ESCALATE",
                    reason="DEADLOCK",
                    detail="EMPTY_TOOLS_PROPOSED",
                    other_reasons=other_reasons,
                )
            return Decision(
                action="CONTINUE",
                tools=cmd.tools,
                reason=None,
                detail=None,
                other_reasons=other_reasons,
            )

    # Fallback escalation for unhandled or unknown command kinds
    return Decision(
        action="ESCALATE",
        reason="DEADLOCK",
        detail=f"UNKNOWN_COMMAND_{cmd.kind}",
        other_reasons=other_reasons,
    )
