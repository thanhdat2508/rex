"""Telemetry Trace Store with PII sanitization and audit schemas (Step B9).

Logs granular turn-by-turn execution records required by reporting pipelines (D, E)
and human reviewer audit dashboards.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import re
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
import uuid


def mask_pii_string(val: str) -> str:
    """Mask sensitive identifiers such as IP addresses, credit cards, and emails."""
    # Mask IPv4: 103.28.32.11 -> 103.x.x.11
    ip_pattern = r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$"
    m_ip = re.match(ip_pattern, val.strip())
    if m_ip:
        return f"{m_ip.group(1)}.x.x.{m_ip.group(4)}"

    # Mask card / account numbers > 8 digits: preserve first and last 4
    if val.isdigit() and len(val) >= 12:
        return f"{val[:4]}****{val[-4:]}"

    # Mask email: user@domain.com -> u***@domain.com
    if "@" in val and "." in val:
        parts = val.split("@", 1)
        name = parts[0]
        domain = parts[1]
        masked_name = name[0] + "***" if len(name) > 1 else name + "***"
        return f"{masked_name}@{domain}"

    return val


def mask_pii(data: Any) -> Any:
    """Recursively sanitize parameters dictionary, replacing PII with masked tokens."""
    if isinstance(data, dict):
        return {k: mask_pii(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [mask_pii(v) for v in data]
    elif isinstance(data, str):
        return mask_pii_string(data)
    return data


@dataclass
class TraceStep:
    """Telemetry item representing an individual tool call or terminal decision."""
    step_no: int
    round: int
    action: str                                  # 'CALL_TOOL', 'VERDICT', 'ESCALATE'
    status: str                                  # 'OK', 'TIMEOUT', 'ERROR', 'FILTERED'
    tool: Optional[str] = None
    tool_impact: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    result: Dict[str, Any] = field(default_factory=dict)
    sufficiency_before: Optional[float] = None
    sufficiency_after: Optional[float] = None
    duration_ms: float = 0.0


@dataclass
class CaseTrace:
    """Complete trace carrier for a case investigation."""
    trace_id: str
    ticket_id: str
    created_at: str
    final_action: Optional[str] = None
    final_reason: Optional[str] = None
    steps: List[TraceStep] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "ticket_id": self.ticket_id,
            "created_at": self.created_at,
            "final_action": self.final_action,
            "final_reason": self.final_reason,
            "steps": [asdict(s) for s in self.steps],
        }


@runtime_checkable
class TraceStore(Protocol):
    """Protocol for telemetry trace logging."""

    def start_trace(self, ticket_id: str) -> str: ...
    def append_step(self, ticket_id: str, step: TraceStep) -> None: ...
    def finish_trace(
        self,
        ticket_id: str,
        final_action: str,
        final_reason: Optional[str] = None,
    ) -> CaseTrace: ...
    def get_trace(self, ticket_id: str) -> Optional[CaseTrace]: ...


class InMemoryTraceStore(TraceStore):
    """In-memory trace store managing sanitized telemetry."""

    def __init__(self) -> None:
        self.traces: Dict[str, CaseTrace] = {}
        self.step_counters: Dict[str, int] = {}

    def start_trace(self, ticket_id: str) -> str:
        trace_id = f"rex-trace-{uuid.uuid4().hex[:8]}"
        self.traces[ticket_id] = CaseTrace(
            trace_id=trace_id,
            ticket_id=ticket_id,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self.step_counters[ticket_id] = 0
        return trace_id

    def append_step(self, ticket_id: str, step: TraceStep) -> None:
        trace = self.traces.get(ticket_id)
        if trace is None:
            self.start_trace(ticket_id)
            trace = self.traces[ticket_id]

        self.step_counters[ticket_id] = self.step_counters.get(ticket_id, 0) + 1
        # Ensure step_no is monotonically incremented
        sanitized_params = mask_pii(step.params)
        sanitized_step = TraceStep(
            step_no=self.step_counters[ticket_id],
            round=step.round,
            action=step.action,
            status=step.status,
            tool=step.tool,
            tool_impact=step.tool_impact,
            params=sanitized_params,
            result=dict(step.result),
            sufficiency_before=step.sufficiency_before,
            sufficiency_after=step.sufficiency_after,
            duration_ms=step.duration_ms,
        )
        trace.steps.append(sanitized_step)

    def finish_trace(
        self,
        ticket_id: str,
        final_action: str,
        final_reason: Optional[str] = None,
    ) -> CaseTrace:
        trace = self.traces.get(ticket_id)
        if trace is None:
            self.start_trace(ticket_id)
            trace = self.traces[ticket_id]

        trace.final_action = final_action
        trace.final_reason = final_reason
        return trace

    def get_trace(self, ticket_id: str) -> Optional[CaseTrace]:
        return self.traces.get(ticket_id)
