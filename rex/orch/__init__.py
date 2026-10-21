"""Rex Orchestration Package (Part B).

The control and execution layer wrapping the Rex decision model: Rex selects, B executes.
"""

from .agent_port import AgentCommand, AgentPort, Case, CaseStatus
from .fake_agent import FakeAgent
from .guard import Guard
from .handoff import HandoffPayload, finish_case
from .metrics import OrchMetrics
from .packer import PackedEvidence, format_metric_value, format_tool_result, pack_evidence
from .params import compute_availability, resolve_param, resolve_tool_args
from .registry import build_tool_specs, load_tools_config
from .rex_adapter import RexAdapter
from .runner import CaseRunner
from .state_store import CaseOutcome, InMemoryStateStore, RoundRecord, StateStore, ToolResult
from .stop import Decision, StopConfig, check_epsilon_stagnation, decide
from .tool_client import ToolCall, ToolClient, ToolSession, validate_metrics
from .tool_runner import ToolRunner
from .trace_store import CaseTrace, InMemoryTraceStore, TraceStep, TraceStore, mask_pii

__all__ = [
    "AgentCommand",
    "AgentPort",
    "Case",
    "CaseStatus",
    "FakeAgent",
    "Guard",
    "HandoffPayload",
    "finish_case",
    "OrchMetrics",
    "PackedEvidence",
    "format_metric_value",
    "format_tool_result",
    "pack_evidence",
    "compute_availability",
    "resolve_param",
    "resolve_tool_args",
    "build_tool_specs",
    "load_tools_config",
    "RexAdapter",
    "CaseRunner",
    "CaseOutcome",
    "InMemoryStateStore",
    "RoundRecord",
    "StateStore",
    "ToolResult",
    "Decision",
    "StopConfig",
    "check_epsilon_stagnation",
    "decide",
    "ToolCall",
    "ToolClient",
    "ToolSession",
    "validate_metrics",
    "ToolRunner",
    "CaseTrace",
    "InMemoryTraceStore",
    "TraceStep",
    "TraceStore",
    "mask_pii",
]
