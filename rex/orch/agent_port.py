"""AgentPort abstraction interface and command contracts for Rex Orchestration (B0).

This module isolates the orchestration engine from specific model execution backends.
Both FakeAgent and RexAdapter implement the AgentPort protocol.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


class CaseStatus(str, Enum):
    """Lifecycle state machine for an investigated case."""
    OPEN = "OPEN"
    INVESTIGATING = "INVESTIGATING"
    DECIDED = "DECIDED"
    ESCALATED = "ESCALATED"


@dataclass
class Case:
    """Incident case carrier containing initial evidence, user state, and ticket metadata."""
    ticket_id: str
    type_token: str
    text: str = ""
    state: Dict[str, Any] = field(default_factory=dict)
    initial_evidence: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentCommand:
    """Command emitted by an agent to the orchestrator at each turn."""
    kind: str                                    # 'CALL_TOOL', 'VERDICT', 'STALL'
    tools: List[str] = field(default_factory=list)
    s: float = 0.0                               # Sufficiency probability
    risk: float = 0.0                            # Risk probability
    verdict_probs: Dict[str, float] = field(default_factory=dict)
    conformal_set: List[str] = field(default_factory=list)
    alive: List[str] = field(default_factory=list)
    trace: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class AgentPort(Protocol):
    """Uniform protocol implemented by FakeAgent and RexAdapter."""

    def start(self, case: Case, available: Dict[str, bool]) -> AgentCommand:
        """Start investigation for a case and return the initial command."""
        ...

    def observe(
        self,
        text: str,
        observations: Dict[str, Any],
        available: Dict[str, bool],
    ) -> AgentCommand:
        """Provide round observations and return next command."""
        ...
