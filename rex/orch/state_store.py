"""State Store interface and in-memory implementation for Rex Orchestration (B2).

Tracks case records, lifecycle transitions, tool outputs, evidence chains,
and case resolution outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Set, runtime_checkable

from .agent_port import Case, CaseStatus


class InvalidStateTransitionError(Exception):
    """Raised when attempting an unauthorized case status transition."""
    pass


@dataclass(frozen=True)
class ToolResult:
    """Standardized representation of a single tool invocation result."""
    tool: str
    status: str                                  # 'OK', 'TIMEOUT', 'ERROR', 'FILTERED'
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    duration_ms: float = 0.0


@dataclass(frozen=True)
class RoundRecord:
    """Checkpoint metadata recorded at the conclusion of each investigation round."""
    round_k: int
    s: float
    n_calls: int
    called_tools: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class CaseOutcome:
    """Final decision or escalation output produced for a case."""
    action: str                                  # 'AUTO_VERDICT' or 'ESCALATE'
    verdict: Optional[str] = None
    verdict_probs: Dict[str, float] = field(default_factory=dict)
    conformal_set: List[str] = field(default_factory=list)
    risk: float = 0.0
    escalation_reason: Optional[str] = None      # HIGH_RISK, DEADLOCK, BUDGET_EXHAUSTED, TOOL_BLOCKED
    detail: Optional[str] = None                 # CONFORMAL_LOW_S, NO_CALLABLE_TOOL, etc.
    trace_id: Optional[str] = None
    decided_at: Optional[str] = None


@runtime_checkable
class StateStore(Protocol):
    """Abstract protocol for case state persistence."""

    def load_case(self, ticket_id: str) -> Case: ...
    def set_status(self, ticket_id: str, status: CaseStatus) -> None: ...
    def append_tool_result(self, ticket_id: str, result: ToolResult) -> None: ...
    def append_evidence(self, ticket_id: str, round_k: int, text: str) -> None: ...
    def record_round(
        self,
        ticket_id: str,
        round_k: int,
        s: float,
        n_calls: int,
        called_tools: List[str],
    ) -> None: ...
    def record_outcome(self, ticket_id: str, outcome: CaseOutcome) -> int: ...


class InMemoryStateStore(StateStore):
    """Thread-safe in-memory implementation of StateStore adhering to AGENTS.md."""

    # Allowed status transitions:
    # OPEN -> INVESTIGATING -> DECIDED or ESCALATED
    VALID_TRANSITIONS: Dict[CaseStatus, Set[CaseStatus]] = {
        CaseStatus.OPEN: {CaseStatus.INVESTIGATING},
        CaseStatus.INVESTIGATING: {CaseStatus.DECIDED, CaseStatus.ESCALATED},
        CaseStatus.DECIDED: set(),
        CaseStatus.ESCALATED: set(),
    }

    def __init__(self) -> None:
        self.cases: Dict[str, Case] = {}
        self.statuses: Dict[str, CaseStatus] = {}
        self.tool_results: Dict[str, List[ToolResult]] = {}
        self.evidence: Dict[str, List[tuple[int, str]]] = {}
        self.rounds: Dict[str, List[RoundRecord]] = {}
        self.outcomes: Dict[str, CaseOutcome] = {}
        self.decision_seqs: Dict[str, int] = {}

    def save_case(self, case: Case) -> None:
        """Register or update initial case record."""
        self.cases[case.ticket_id] = case
        if case.ticket_id not in self.statuses:
            self.statuses[case.ticket_id] = CaseStatus.OPEN
            self.tool_results[case.ticket_id] = []
            self.evidence[case.ticket_id] = []
            self.rounds[case.ticket_id] = []
            self.decision_seqs[case.ticket_id] = 0

    def load_case(self, ticket_id: str) -> Case:
        if ticket_id not in self.cases:
            raise KeyError(f"Case with ticket_id '{ticket_id}' not found in StateStore")
        return self.cases[ticket_id]

    def get_status(self, ticket_id: str) -> CaseStatus:
        if ticket_id not in self.statuses:
            raise KeyError(f"Case with ticket_id '{ticket_id}' not found")
        return self.statuses[ticket_id]

    def set_status(self, ticket_id: str, status: CaseStatus) -> None:
        current = self.get_status(ticket_id)
        if current == status:
            return

        allowed = self.VALID_TRANSITIONS.get(current, set())
        if status not in allowed:
            raise InvalidStateTransitionError(
                f"Cannot transition case '{ticket_id}' from {current} to {status}. "
                f"Allowed transitions: {allowed}"
            )
        self.statuses[ticket_id] = status

    def append_tool_result(self, ticket_id: str, result: ToolResult) -> None:
        if ticket_id not in self.cases:
            raise KeyError(f"Case with ticket_id '{ticket_id}' not found")
        self.tool_results.setdefault(ticket_id, []).append(result)

    def append_evidence(self, ticket_id: str, round_k: int, text: str) -> None:
        if ticket_id not in self.cases:
            raise KeyError(f"Case with ticket_id '{ticket_id}' not found")
        self.evidence.setdefault(ticket_id, []).append((round_k, text))

    def record_round(
        self,
        ticket_id: str,
        round_k: int,
        s: float,
        n_calls: int,
        called_tools: List[str],
    ) -> None:
        if ticket_id not in self.cases:
            raise KeyError(f"Case with ticket_id '{ticket_id}' not found")
        self.rounds.setdefault(ticket_id, []).append(
            RoundRecord(round_k=round_k, s=s, n_calls=n_calls, called_tools=list(called_tools))
        )

    def record_outcome(self, ticket_id: str, outcome: CaseOutcome) -> int:
        if ticket_id not in self.cases:
            raise KeyError(f"Case with ticket_id '{ticket_id}' not found")
        current_seq = self.decision_seqs.get(ticket_id, 0) + 1
        self.decision_seqs[ticket_id] = current_seq
        self.outcomes[ticket_id] = outcome
        return current_seq

    def get_evidence_chain(self, ticket_id: str) -> List[str]:
        """Return sequential list of evidence texts recorded across rounds."""
        return [text for _, text in self.evidence.get(ticket_id, [])]

    def get_called_tools(self, ticket_id: str) -> List[str]:
        """Return list of distinct tool names executed for this case."""
        seen: List[str] = []
        for r in self.tool_results.get(ticket_id, []):
            if r.tool not in seen:
                seen.append(r.tool)
        return seen

    def get_tool_results(self, ticket_id: str) -> List[ToolResult]:
        return list(self.tool_results.get(ticket_id, []))

    def get_outcome(self, ticket_id: str) -> Optional[CaseOutcome]:
        return self.outcomes.get(ticket_id)
