"""Guardrail access control and post-forward tool filtering for Rex Orchestration (B6).

Guarantees no unauthorized (LIGHT/HEAVY) or masked tools reach execution.
Enforces double-gating: once before model inference and once after model forward.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple


class Guard:
    """Security and authorization gatekeeper for tool dispatch."""

    def __init__(
        self,
        allowed_levels: Optional[Set[str]] = None,
        high_risk_mode: bool = False,
    ) -> None:
        # By default, automated orchestration strictly allows READ tools only
        self.allowed_levels: Set[str] = set(allowed_levels) if allowed_levels is not None else {"READ"}
        self.high_risk_mode: bool = high_risk_mode

    def is_allowed(self, tool_name: str, level: str = "READ") -> bool:
        """Pre-forward check: returns False if tool level is unauthorized."""
        if self.high_risk_mode and level != "READ":
            return False
        return level in self.allowed_levels

    def filter_selected_tools(
        self,
        selected_tools: List[str],
        tools_cfg: Dict[str, Any],
        available_mask: Dict[str, bool],
        called_tools: Optional[List[str]] = None,
    ) -> Tuple[List[str], Optional[str], Optional[str]]:
        """Post-forward filter: verifies tools proposed by the model.

        Returns (approved_tools, failure_reason, failure_detail).
        failure_reason may be 'DEADLOCK' or 'TOOL_BLOCKED'.
        """
        if not selected_tools:
            return [], None, None

        cfg_tools = tools_cfg.get("tools", {})
        approved: List[str] = []
        already_called = set(called_tools or [])

        prohibited_level_detected = False
        unresolved_params_detected = False

        for name in selected_tools:
            if name in already_called:
                # Anti-repetition: do not re-run tool already completed
                continue

            conf = cfg_tools.get(name, {})
            level = conf.get("level", "READ")

            # Check level authorization
            if not self.is_allowed(name, level):
                prohibited_level_detected = True
                continue

            # Check parameter availability
            if not available_mask.get(name, False):
                unresolved_params_detected = True
                continue

            approved.append(name)

        if not approved:
            if prohibited_level_detected:
                return [], "TOOL_BLOCKED", "PROHIBITED_LEVEL"
            if unresolved_params_detected:
                return [], "DEADLOCK", "NO_CALLABLE_TOOL"
            # All tools were duplicates
            return [], "DEADLOCK", "DUPLICATE_CALLS_SUPPRESSED"

        return approved, None, None
