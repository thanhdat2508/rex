"""MCP Tool Client with concurrent execution, timeouts, and metric validation (B4).

Manages per-server connection limits, enforces execution deadlines,
rejects HEAVY tools, and validates observations against MetricSpec schemas.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from rex.rex_registry import MetricSpec, ToolSpec

from .state_store import ToolResult


@dataclass(frozen=True)
class ToolCall:
    """Execution manifest for an individual tool invocation."""
    tool: str
    server: str
    remote_name: str
    args: Dict[str, Any]
    timeout_s: float
    level: str = "READ"


@runtime_checkable
class ToolSession(Protocol):
    """Protocol for an MCP or mock tool session."""
    async def call_tool(
        self,
        remote_name: str,
        args: Dict[str, Any],
        timeout_s: float,
    ) -> Dict[str, Any]:
        ...


def validate_metrics(
    tool_name: str,
    raw_output: Any,
    tool_spec: Optional[ToolSpec] = None,
    metric_specs: Optional[Dict[str, MetricSpec]] = None,
) -> tuple[Dict[str, Any], List[str]]:
    """Validate raw tool response dictionary against tool returns and metric definitions.

    Returns (valid_metrics_dict, errors_list).
    Discards missing, mistyped, or out-of-range metrics.
    """
    valid_metrics: Dict[str, Any] = {}
    errors: List[str] = []

    if not isinstance(raw_output, dict):
        return {}, [f"Tool '{tool_name}' returned non-dict output of type {type(raw_output).__name__}"]

    expected_metrics = tool_spec.returns if tool_spec is not None else list(raw_output.keys())

    for metric_name in expected_metrics:
        if metric_name not in raw_output:
            errors.append(f"Missing expected metric '{metric_name}' in output")
            continue

        val = raw_output[metric_name]
        mspec = metric_specs.get(metric_name) if metric_specs is not None else None

        if mspec is None:
            # No metric specification; accept as-is
            valid_metrics[metric_name] = val
            continue

        # Validate type and bounds
        kind = mspec.kind
        try:
            if kind == "bool":
                if isinstance(val, bool):
                    checked_val: Any = val
                elif isinstance(val, (int, float)) and val in (0, 1):
                    checked_val = bool(val)
                elif isinstance(val, str) and val.lower() in ("true", "false", "0", "1"):
                    checked_val = val.lower() in ("true", "1")
                else:
                    errors.append(f"Metric '{metric_name}' expected bool, got {type(val).__name__} ({val})")
                    continue
            elif kind == "int":
                if isinstance(val, bool):
                    errors.append(f"Metric '{metric_name}' expected int, got bool ({val})")
                    continue
                checked_val = int(val)
                if mspec.min_val is not None and checked_val < mspec.min_val:
                    errors.append(f"Metric '{metric_name}' value {checked_val} < min_val {mspec.min_val}")
                    continue
                if mspec.max_val is not None and checked_val > mspec.max_val:
                    errors.append(f"Metric '{metric_name}' value {checked_val} > max_val {mspec.max_val}")
                    continue
            elif kind == "float":
                if isinstance(val, bool):
                    errors.append(f"Metric '{metric_name}' expected float, got bool ({val})")
                    continue
                checked_val = float(val)
                if mspec.min_val is not None and checked_val < mspec.min_val:
                    errors.append(f"Metric '{metric_name}' value {checked_val} < min_val {mspec.min_val}")
                    continue
                if mspec.max_val is not None and checked_val > mspec.max_val:
                    errors.append(f"Metric '{metric_name}' value {checked_val} > max_val {mspec.max_val}")
                    continue
            else:
                checked_val = val

            valid_metrics[metric_name] = checked_val

        except (ValueError, TypeError) as e:
            errors.append(f"Metric '{metric_name}' validation error: {e}")

    return valid_metrics, errors


class ToolClient:
    """Orchestration client for executing tool calls across servers concurrently."""

    def __init__(
        self,
        sessions: Optional[Dict[str, ToolSession]] = None,
        max_concurrency_per_server: int = 5,
        tool_specs: Optional[Dict[str, ToolSpec]] = None,
        metric_specs: Optional[Dict[str, MetricSpec]] = None,
    ) -> None:
        self.sessions: Dict[str, ToolSession] = dict(sessions) if sessions is not None else {}
        self.semaphores: Dict[str, asyncio.Semaphore] = {}
        self.max_concurrency = max_concurrency_per_server
        self.tool_specs = dict(tool_specs) if tool_specs is not None else {}
        self.metric_specs = dict(metric_specs) if metric_specs is not None else {}

    def get_semaphore(self, server: str) -> asyncio.Semaphore:
        if server not in self.semaphores:
            self.semaphores[server] = asyncio.Semaphore(self.max_concurrency)
        return self.semaphores[server]

    def register_session(self, server: str, session: ToolSession) -> None:
        self.sessions[server] = session

    async def call_single_tool(
        self,
        c: ToolCall,
        deadline: float,
    ) -> ToolResult:
        # Guardrail check: NEVER execute HEAVY tools automatically
        if c.level == "HEAVY":
            return ToolResult(
                tool=c.tool,
                status="FILTERED",
                metrics={},
                errors=["HEAVY tools are not permitted for automated execution"],
                duration_ms=0.0,
            )

        sem = self.get_semaphore(c.server)
        now = time.monotonic()
        remaining_budget = deadline - now
        if remaining_budget <= 0:
            return ToolResult(
                tool=c.tool,
                status="TIMEOUT",
                metrics={},
                errors=["Case deadline exceeded prior to dispatch"],
                duration_ms=0.0,
            )

        timeout = min(c.timeout_s, remaining_budget)
        session = self.sessions.get(c.server)
        if session is None:
            return ToolResult(
                tool=c.tool,
                status="ERROR",
                metrics={},
                errors=[f"No active session configured for server '{c.server}'"],
                duration_ms=0.0,
            )

        t0 = time.monotonic()
        try:
            async with sem:
                raw_res = await asyncio.wait_for(
                    session.call_tool(c.remote_name, c.args, timeout_s=timeout),
                    timeout=timeout,
                )
                duration_ms = (time.monotonic() - t0) * 1000.0
                spec = self.tool_specs.get(c.tool)
                metrics, errors = validate_metrics(c.tool, raw_res, spec, self.metric_specs)
                status = "OK" if not errors or len(metrics) > 0 else "ERROR"
                return ToolResult(
                    tool=c.tool,
                    status=status,
                    metrics=metrics,
                    errors=errors,
                    duration_ms=duration_ms,
                )
        except asyncio.TimeoutError:
            duration_ms = (time.monotonic() - t0) * 1000.0
            return ToolResult(
                tool=c.tool,
                status="TIMEOUT",
                metrics={},
                errors=[f"Execution timed out after {timeout:.3f}s"],
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.monotonic() - t0) * 1000.0
            return ToolResult(
                tool=c.tool,
                status="ERROR",
                metrics={},
                errors=[f"{type(e).__name__}: {e}"],
                duration_ms=duration_ms,
            )

    async def call_tools(
        self,
        calls: List[ToolCall],
        deadline: float,
    ) -> List[ToolResult]:
        """Execute a list of tool calls in parallel with individual error isolation."""
        if not calls:
            return []
        tasks = [self.call_single_tool(c, deadline) for c in calls]
        results = await asyncio.gather(*tasks)
        return list(results)
