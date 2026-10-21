"""FakeAgent for mock multi-round orchestration replay (B0).

Allows end-to-end execution of orchestration pipelines without a neural network or GPU.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .agent_port import AgentCommand, AgentPort, Case


class FakeAgent(AgentPort):
    """Scriptable mock agent replaying predetermined commands turn by turn."""

    def __init__(self, commands: Optional[List[AgentCommand]] = None) -> None:
        self.commands: List[AgentCommand] = list(commands) if commands is not None else []
        self.current_step: int = 0
        self.observed_steps: List[Dict[str, Any]] = []
        self.active_case: Optional[Case] = None

    def start(self, case: Case, available: Dict[str, bool]) -> AgentCommand:
        self.active_case = case
        self.current_step = 0
        self.observed_steps = []
        if self.current_step < len(self.commands):
            cmd = self.commands[self.current_step]
            self.current_step += 1
            return cmd
        return AgentCommand(kind="VERDICT", s=1.0)

    def observe(
        self,
        text: str,
        observations: Dict[str, Any],
        available: Dict[str, bool],
    ) -> AgentCommand:
        self.observed_steps.append({
            "text": text,
            "observations": observations,
            "available": dict(available),
        })
        if self.current_step < len(self.commands):
            cmd = self.commands[self.current_step]
            self.current_step += 1
            return cmd
        return AgentCommand(kind="VERDICT", s=1.0)

    @classmethod
    def case_anh_a(cls) -> FakeAgent:
        """Standard scenario 'vụ anh A' used across tests:
        Round 1: CALL_TOOL (geo_ip_lookup, device_fingerprint) at S=0.40
        Round 2: CALL_TOOL (auth_history) at S=0.62
        Round 3: VERDICT (H_ato) at S=0.91
        """
        return cls(commands=[
            AgentCommand(
                kind="CALL_TOOL",
                tools=["geo_ip_lookup", "device_fingerprint"],
                s=0.40,
                risk=0.15,
                trace={"round": 1, "sufficiency": 0.40},
            ),
            AgentCommand(
                kind="CALL_TOOL",
                tools=["auth_history"],
                s=0.62,
                risk=0.20,
                trace={"round": 2, "sufficiency": 0.62},
            ),
            AgentCommand(
                kind="VERDICT",
                tools=[],
                s=0.91,
                risk=0.10,
                verdict_probs={"H_ato": 0.88, "H_travel": 0.08, "H_novel": 0.04},
                conformal_set=["H_ato"],
                alive=["H_ato"],
                trace={"round": 3, "sufficiency": 0.91, "winner": "H_ato"},
            ),
        ])
