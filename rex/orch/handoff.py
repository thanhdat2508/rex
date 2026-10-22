"""Handoff protocol to Collaborator C (Decision Adapter) (Step B11).

Generates idempotent decision payloads, updates StateStore to terminal status,
and delivers auditable escalation bundles to human reviewers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .agent_port import AgentCommand, Case, CaseStatus
from .state_store import CaseOutcome, StateStore
from .stop import Decision
from .trace_store import TraceStore


@dataclass(frozen=True)
class HandoffPayload:
    """Standardized decision output delivered to Collaborator C."""
    request_id: str                              # ticket_id:decision_seq
    ticket_id: str
    decision_seq: int
    outcome: str                                 # 'AUTO_VERDICT' or 'ESCALATE'
    verdict: Optional[str]
    verdict_probs: Dict[str, float]
    conformal_set: List[str]
    risk: float
    escalation_reason: Optional[str]
    detail: Optional[str]
    decided_at: str
    trace_id: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def finish_case(
    case: Case,
    decision: Decision,
    cmd: AgentCommand,
    store: StateStore,
    trace_store: TraceStore,
) -> HandoffPayload:
    """Conclude case, finalize trace, record outcome, and package handoff payload."""
    ticket_id = case.ticket_id

    # 1. Finalize trace first
    trace = trace_store.finish_trace(
        ticket_id=ticket_id,
        final_action=decision.action,
        final_reason=decision.reason,
    )

    # 2. Derive winning hypothesis label if AUTO_VERDICT
    verdict: Optional[str] = None
    if decision.action == "AUTO_VERDICT":
        if cmd.conformal_set and len(cmd.conformal_set) == 1:
            verdict = cmd.conformal_set[0]
        elif cmd.verdict_probs:
            verdict = max(cmd.verdict_probs.items(), key=lambda kv: kv[1])[0]
        elif cmd.alive and len(cmd.alive) == 1:
            verdict = cmd.alive[0]
        else:
            verdict = "H_novel"
    else:
        # Escalated cases do NOT publish an automated verdict
        verdict = None

    decided_at = datetime.now(timezone.utc).isoformat()
    outcome = CaseOutcome(
        action=decision.action,
        verdict=verdict,
        verdict_probs=dict(cmd.verdict_probs),
        conformal_set=list(cmd.conformal_set),
        risk=cmd.risk,
        escalation_reason=decision.reason,
        detail=decision.detail,
        trace_id=trace.trace_id,
        decided_at=decided_at,
    )

    # 3. Transition case to terminal status in StateStore
    new_status = CaseStatus.DECIDED if decision.action == "AUTO_VERDICT" else CaseStatus.ESCALATED
    store.set_status(ticket_id, new_status)

    # 4. Record outcome and retrieve monotonically increasing decision_seq
    decision_seq = store.record_outcome(ticket_id, outcome)
    request_id = f"{ticket_id}:{decision_seq}"

    return HandoffPayload(
        request_id=request_id,
        ticket_id=ticket_id,
        decision_seq=decision_seq,
        outcome=decision.action,
        verdict=verdict,
        verdict_probs=dict(cmd.verdict_probs),
        conformal_set=list(cmd.conformal_set),
        risk=cmd.risk,
        escalation_reason=decision.reason,
        detail=decision.detail,
        decided_at=decided_at,
        trace_id=trace.trace_id,
    )
