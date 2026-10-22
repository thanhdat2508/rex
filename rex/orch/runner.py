"""Multi-round Case Runner executing complete incident investigation loops (Step B10).

Orchestrates AgentPort commands, Guard safety checks, ToolRunner execution,
Evidence packing, StateStore persistence, and Collaborator C handoff.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

from .agent_port import AgentPort, CaseStatus
from .guard import Guard
from .handoff import HandoffPayload, finish_case
from .packer import pack_evidence
from .params import compute_availability
from .state_store import StateStore
from .stop import StopConfig, decide
from .tool_runner import ToolRunner
from .trace_store import TraceStep, TraceStore


class CaseRunner:
    """Orchestration driver executing multi-round investigations for a ticket."""

    def __init__(
        self,
        agent: AgentPort,
        tool_runner: ToolRunner,
        store: StateStore,
        trace_store: TraceStore,
        model_tools: Sequence[str],
        tools_cfg: Dict[str, Any],
        stop_cfg: Optional[StopConfig] = None,
        guard: Optional[Guard] = None,
        tok: Optional[Any] = None,
        default_deadline_s: float = 5.0,
    ) -> None:
        self.agent = agent
        self.tool_runner = tool_runner
        self.store = store
        self.trace_store = trace_store
        self.model_tools = list(model_tools)
        self.tools_cfg = tools_cfg
        self.stop_cfg = stop_cfg or StopConfig()
        self.guard = guard or Guard()
        self.tok = tok
        self.default_deadline_s = default_deadline_s

    async def run_case(
        self,
        ticket_id: str,
        timeout_s: Optional[float] = None,
    ) -> HandoffPayload:
        """Execute full investigation lifecycle for a registered case."""
        case = self.store.load_case(ticket_id)
        self.store.set_status(ticket_id, CaseStatus.INVESTIGATING)

        deadline = time.monotonic() + (timeout_s or self.default_deadline_s)
        self.trace_store.start_trace(ticket_id)

        round_k = 1
        n_calls = 0
        s_history: List[float] = []
        accumulated_results: Dict[str, Dict[str, Any]] = {}
        called_tools: List[str] = []
        is_high_risk = False

        # 1. Round 1: Compute availability and start agent
        avail_mask = compute_availability(
            self.model_tools,
            self.tools_cfg,
            case,
            accumulated_results,
            guard=self.guard,
        )
        cmd = self.agent.start(case, avail_mask)

        while True:
            s_history.append(cmd.s)
            time_left = deadline - time.monotonic()

            # 2. Check stopping decision
            d = decide(
                cmd=cmd,
                s_history=s_history,
                n_calls=n_calls,
                round_k=round_k,
                time_left_s=time_left,
                high_risk=is_high_risk,
                cfg=self.stop_cfg,
                guard=self.guard,
                tools_cfg=self.tools_cfg,
                available_mask=avail_mask,
                called_tools=called_tools,
            )

            if d.action != "CONTINUE":
                # Finalize investigation
                return finish_case(
                    case=case,
                    decision=d,
                    cmd=cmd,
                    store=self.store,
                    trace_store=self.trace_store,
                )

            # 3. Execute approved tool calls
            s_before = cmd.s
            round_results = await self.tool_runner.run(
                tools=d.tools,
                round_k=round_k,
                deadline=deadline,
                case=case,
                accumulated_results=accumulated_results,
            )

            # Record round results and trace steps
            for r in round_results:
                called_tools.append(r.tool)
                if r.status == "OK":
                    accumulated_results[r.tool] = r.metrics

                tool_conf = self.tools_cfg.get("tools", {}).get(r.tool, {})
                self.trace_store.append_step(
                    ticket_id,
                    TraceStep(
                        step_no=0,  # will be auto-incremented by trace_store
                        round=round_k,
                        action="CALL_TOOL",
                        status=r.status,
                        tool=r.tool,
                        tool_impact=tool_conf.get("level", "READ"),
                        params=tool_conf.get("params", {}),
                        result=r.metrics,
                        sufficiency_before=s_before,
                        sufficiency_after=None,  # will be updated with next cmd.s
                        duration_ms=r.duration_ms,
                    ),
                )

            n_calls += len(d.tools)

            # 4. Pack round observations into bounded 64-token evidence chunk
            packed_ev = pack_evidence(round_results, round_k=round_k + 1, tok=self.tok)
            self.store.append_evidence(ticket_id, round_k, packed_ev.text)
            self.store.record_round(ticket_id, round_k, cmd.s, n_calls, list(called_tools))

            round_k += 1

            # 5. Provide feedback to agent
            avail_mask = compute_availability(
                self.model_tools,
                self.tools_cfg,
                case,
                accumulated_results,
                guard=self.guard,
            )
            cmd = self.agent.observe(
                text=packed_ev.text,
                observations=packed_ev.observations,
                available=avail_mask,
            )
