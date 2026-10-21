"""Tool Runner coordinating parameter extraction, client dispatch, and state recording (B5).

Single point of execution transforming agent CALL_TOOL commands into validated ToolResults.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .agent_port import Case
from .params import resolve_tool_args
from .state_store import StateStore, ToolResult
from .tool_client import ToolCall, ToolClient


class ToolRunner:
    """Coordinates execution of tools for an investigation round."""

    def __init__(
        self,
        client: ToolClient,
        tools_cfg: Dict[str, Any],
        store: StateStore,
        guard: Optional[Any] = None,
    ) -> None:
        self.client = client
        self.tools_cfg = tools_cfg
        self.store = store
        self.guard = guard

    async def run(
        self,
        tools: List[str],
        round_k: int,
        deadline: float,
        case: Case,
        accumulated_results: Dict[str, Dict[str, Any]],
    ) -> List[ToolResult]:
        """Resolve parameters, dispatch tool calls, and append results to StateStore."""
        if not tools:
            return []

        calls: List[ToolCall] = []
        filtered_results: List[ToolResult] = []

        cfg_tools = self.tools_cfg.get("tools", {})
        for tool_name in tools:
            # 1. Guard check if guard provided
            if self.guard is not None:
                if hasattr(self.guard, "filter_tool"):
                    allowed, reason = self.guard.filter_tool(tool_name)
                    if not allowed:
                        res = ToolResult(
                            tool=tool_name,
                            status="FILTERED",
                            metrics={},
                            errors=[f"Guard rejected tool: {reason}"],
                            duration_ms=0.0,
                        )
                        filtered_results.append(res)
                        continue

            tool_conf = cfg_tools.get(tool_name, {})
            args = resolve_tool_args(tool_name, self.tools_cfg, case, accumulated_results)
            if args is None:
                # Required parameter missing
                res = ToolResult(
                    tool=tool_name,
                    status="ERROR",
                    metrics={},
                    errors=["Unable to resolve required parameters for invocation"],
                    duration_ms=0.0,
                )
                filtered_results.append(res)
                continue

            timeout_ms = float(tool_conf.get("timeout_ms", 100))
            calls.append(
                ToolCall(
                    tool=tool_name,
                    server=str(tool_conf.get("server", "default")),
                    remote_name=str(tool_conf.get("remote_name", tool_name)),
                    args=args,
                    timeout_s=timeout_ms / 1000.0,
                    level=str(tool_conf.get("level", "READ")),
                )
            )

        executed_results = await self.client.call_tools(calls, deadline=deadline)
        all_results = filtered_results + executed_results

        # Record all tool outputs in StateStore
        for res in all_results:
            self.store.append_tool_result(case.ticket_id, res)

        return all_results
