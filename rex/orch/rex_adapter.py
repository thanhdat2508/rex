"""RexAdapter bridging RexDecisionModel to the AgentPort protocol (Step B12).

Wraps RexDecisionModel.forward(), HybridPoEEngine falsification,
and HardMaskingManager to execute live neural models inside the orchestrator.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from rex.rex_format import collate_rex, encode_evidence, encode_rex_input
from rex.rex_poe import HardMaskingManager, HybridPoEEngine
from rex.rex_registry import ToolSpec
from rex.rex_types import RexState, RexStepOutput

from .agent_port import AgentCommand, AgentPort, Case


class RexAdapter(AgentPort):
    """Adapter implementing AgentPort over a neural RexDecisionModel instance."""

    def __init__(
        self,
        model: Any,
        tok: Any,
        instruction: str,
        hypotheses: Dict[str, str],
        rules: Optional[Dict[str, List[Any]]] = None,
        tool_hyp_mapping: Optional[Dict[str, List[str]]] = None,
        tool_specs: Optional[Sequence[ToolSpec]] = None,
        theta_elim: float = 0.20,
    ) -> None:
        self.model = model
        self.tok = tok
        self.instruction = instruction
        self.hypotheses = dict(hypotheses)
        self.hyp_keys = list(hypotheses.keys())
        self.rules = dict(rules) if rules is not None else {}
        self.tool_hyp_mapping = dict(tool_hyp_mapping) if tool_hyp_mapping is not None else {}
        self.tool_specs = list(tool_specs) if tool_specs is not None else None
        self.theta_elim = theta_elim

        self.poe_engine = HybridPoEEngine(default_theta_elim=theta_elim)
        self.mask_manager = HardMaskingManager()

        self.state: Optional[RexState] = None
        self.alive: Dict[str, bool] = {k: True for k in self.hyp_keys}
        self.called_tools: List[str] = []

    @property
    def device(self) -> torch.device:
        if hasattr(self.model, "tool_embeddings"):
            return self.model.tool_embeddings.device
        elif hasattr(self.model, "device"):
            return self.model.device
        return torch.device("cpu")

    def _combine_tool_masks(self, available: Dict[str, bool]) -> List[bool]:
        """Combine Part A hard falsification masking with Part B environment availability."""
        general_tools = [
            spec.name for spec in (self.tool_specs or []) if getattr(spec, "general", False)
        ]
        if not general_tools and hasattr(self.model, "tool_meta"):
            general_tools = [t for t, meta in self.model.tool_meta.items() if meta.get("general", False)]
        mask_a = self.mask_manager.compute_tool_mask(
            self.model.tools,
            self.alive,
            self.tool_hyp_mapping,
            called_tools=self.called_tools,
            general_tools=general_tools,
            allow_unmapped=False,
        )
        return [m_a and available.get(tool_name, True) for m_a, tool_name in zip(mask_a, self.model.tools)]

    def to_command(self, out: RexStepOutput) -> AgentCommand:
        """Convert a neural RexStepOutput into an AgentCommand."""
        kind = out.proposed_action[0] if out.proposed_action else "CALL_TOOL"
        tools = out.selected_tools[0] if out.selected_tools else []

        s = float(out.s_prob[0].item()) if hasattr(out.s_prob[0], "item") else float(out.s_prob[0])
        risk = float(out.r_prob[0].item()) if hasattr(out.r_prob[0], "item") else float(out.r_prob[0])

        probs_t = out.verdict_probs[0]
        probs_list = probs_t.tolist() if hasattr(probs_t, "tolist") else list(probs_t)
        verdict_probs = {k: float(v) for k, v in zip(self.hyp_keys, probs_list)}

        conformal = out.conformal_set[0] if out.conformal_set else []
        alive_list = [k for k, v in self.alive.items() if v]
        trace_dict = out.to_trace_dict(0) if hasattr(out, "to_trace_dict") else {}

        return AgentCommand(
            kind=kind,
            tools=tools,
            s=s,
            risk=risk,
            verdict_probs=verdict_probs,
            conformal_set=conformal,
            alive=alive_list,
            trace=trace_dict,
        )

    def start(self, case: Case, available: Dict[str, bool]) -> AgentCommand:
        """Execute turn 1 encoding and model forward pass."""
        # Inference: no dropout and the fixed ponder depth (train mode samples T in [2, 4])
        self.model.eval()
        self.state = None
        self.alive = {k: True for k in self.hyp_keys}
        self.called_tools = []
        if case.state:
            # Metrics known before any tool call (the alert) can already falsify hypotheses
            self.alive = self.poe_engine.evaluate(
                neural_scores=None,
                hyp_keys=self.hyp_keys,
                observations=case.state,
                rules=self.rules,
                prev_alive=self.alive,
            )

        dev = self.device
        tool_mask_list = self._combine_tool_masks(available)
        tool_mask_tensor = torch.tensor([tool_mask_list], dtype=torch.bool, device=dev)

        alive_mask_list = [self.alive.get(k, True) for k in self.hyp_keys]
        alive_mask_tensor = torch.tensor([alive_mask_list], dtype=torch.bool, device=dev)

        encoded = encode_rex_input(
            self.tok,
            type_token=case.type_token,
            instruction=self.instruction,
            hypotheses=self.hypotheses,
            state=case.text,
            alive=self.alive,
            tool_mask=tool_mask_list,
        )
        pad_id = getattr(self.tok, "pad_token_id", 0) or 0
        batch = collate_rex([encoded], pad_token_id=pad_id)
        batch_dev = {k: v.to(dev) for k, v in batch.items()}

        with torch.no_grad():
            _, self.state, step_out = self.model.forward(
                input_ids=batch_dev["input_ids"],
                attention_mask=batch_dev["attention_mask"],
                marker_pos=batch_dev["marker_pos"],
                marker_mask=batch_dev["marker_mask"],
                qtype=batch_dev.get("qtype"),
                tool_mask=tool_mask_tensor,
                hyp_alive=alive_mask_tensor,
                tool_specs=self.tool_specs,
                hyp_names=self.hyp_keys,
            )

        cmd = self.to_command(step_out)
        self.called_tools.extend(cmd.tools)
        return cmd

    def observe(
        self,
        text: str,
        observations: Dict[str, Any],
        available: Dict[str, bool],
    ) -> AgentCommand:
        """Evaluate Popperian elimination, encode evidence, and execute subsequent forward."""
        if self.state is None:
            raise RuntimeError("RexAdapter.observe called before start()")

        dev = self.device

        # 1. Encode 64-token bounded evidence and append it to memory
        ev = encode_evidence(self.tok, text, round_idx=self.state.round_idx + 1, max_tokens=64)
        ev_ids = torch.tensor([ev["input_ids"]], dtype=torch.long, device=dev)
        ev_mask = torch.tensor([ev["attention_mask"]], dtype=torch.bool, device=dev)

        with torch.no_grad():
            self.state = self.model.ingest_evidence(
                self.state, ev_ids, ev_mask, is_truncated=bool(ev.get("is_truncated", False))
            )
            # 2. Neural hypothesis scoring on the updated memory and Popperian elimination
            neural_scores = self.model.hyp_scorer(
                self.state.hyp_markers, self.state.M, self.state.memory_mask
            )

        self.alive = self.poe_engine.evaluate(
            neural_scores=neural_scores,
            hyp_keys=self.hyp_keys,
            observations=observations,
            rules=self.rules,
            theta_elim=self.theta_elim,
            prev_alive=self.alive,
        )

        tool_mask_list = self._combine_tool_masks(available)
        tool_mask_tensor = torch.tensor([tool_mask_list], dtype=torch.bool, device=dev)

        alive_mask_list = [self.alive.get(k, True) for k in self.hyp_keys]
        alive_mask_tensor = torch.tensor([alive_mask_list], dtype=torch.bool, device=dev)

        # 3. Subsequent model forward pass
        with torch.no_grad():
            _, self.state, step_out = self.model.forward(
                state=self.state,
                tool_specs=self.tool_specs,
                tool_mask=tool_mask_tensor,
                hyp_alive=alive_mask_tensor,
                hyp_names=self.hyp_keys,
            )

        cmd = self.to_command(step_out)
        self.called_tools.extend(cmd.tools)
        return cmd
