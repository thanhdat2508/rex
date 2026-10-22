"""InvestigatorAgent: Autonomous multi-round investigative inquiry orchestrator (Step A10).

This module separates procedural environment loops, tool dispatch, and observation intake
from the neural RexDecisionModel inference architecture.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch

from .rex_format import collate_rex, encode_evidence, encode_rex_input, format_observations
from .rex_poe import (
    HardMaskingManager,
    HybridPoEEngine,
    InvestigationResult,
    RoundTrace,
    TerminationReason,
)
from .rex_registry import ToolSpec
from .rex_tools import DEFAULT_R_THRESHOLD, DEFAULT_S_THRESHOLD
from .rex_types import RexState, RexStepOutput


class InvestigatorAgent:
    """Autonomous investigation agent managing multi-round inquiry, tool calls, and Popperian pruning.

    Responsibilities:
        - Orchestrates multi-round agent loop with RexDecisionModel.
        - Translates environment observations into bounded 64-token evidence chunks.
        - Applies Hybrid Popperian elimination (symbolic falsification + neural hypothesis scoring).
        - Enforces early stopping (single survivor, all cataloged eliminated, budget caps).
    """

    def __init__(
        self,
        model: Any,
        theta_elim: float = 0.20,
    ):
        self.model = model
        self.theta_elim = theta_elim
        self.poe_engine = HybridPoEEngine(default_theta_elim=theta_elim)
        self.mask_manager = HardMaskingManager()

    def investigate(
        self,
        tok: Any,
        type_token: str,
        instruction: str,
        hypotheses: Dict[str, str],
        rules: Dict[str, List[Any]],
        tool_hyp_mapping: Dict[str, List[str]],
        tool_dispatcher: Callable[[List[str]], Dict[str, Any]],
        initial_state: Union[str, dict, list] = "",
        max_rounds: int = 3,
        tool_specs: Optional[Sequence[ToolSpec]] = None,
        initial_observations: Optional[Dict[str, Any]] = None,
        stop_on_single_survivor: bool = False,
        **kwargs: Any,
    ) -> InvestigationResult:
        """Execute autonomous multi-round investigation loop.

        Args:
            tok: Tokenizer instance.
            type_token: Question domain type token.
            instruction: Incident prompt / problem description.
            hypotheses: Dictionary mapping hypothesis IDs to textual definitions.
            rules: Falsification rules per hypothesis ID.
            tool_hyp_mapping: Mapping of tool names to supported hypotheses.
            tool_dispatcher: Callback executing approved tool names and returning observations, preferably
                grouped per tool ({tool: {metric: value}}): that renders the same evidence text the
                trainer and Part B use. A flat {metric: value} dict is also accepted.
            initial_state: Initial state or incident metadata string/dict.
            max_rounds: Maximum inquiry budget (rounds).
            tool_specs: Optional rich ToolSpec catalogue.
            initial_observations: Metrics already known before any tool call (the alert). Falsification
                rules are applied to them before round 1, as the training rollouts do. Defaults to
                `initial_state` when that is a dict.
            stop_on_single_survivor: Stop as soon as one cataloged hypothesis survives. When False the
                model decides (VERDICT once S is sufficient), like the Part B orchestrator.

        Returns:
            InvestigationResult with winner hypothesis, probabilities, traces, and escalation status.
        """
        # Inference: no dropout and the fixed ponder depth (train mode samples T in [2, 4])
        self.model.eval()
        hyp_keys = list(hypotheses.keys())
        current_alive = {k: True for k in hyp_keys}
        if initial_observations is None and isinstance(initial_state, dict):
            initial_observations = initial_state
        if initial_observations:
            # Alert metrics can already falsify hypotheses (rollouts apply registry.falsify to the alert)
            current_alive = self.poe_engine.evaluate(
                neural_scores=None,
                hyp_keys=hyp_keys,
                observations=initial_observations,
                rules=rules,
                prev_alive=current_alive,
            )
        state_t = initial_state if isinstance(initial_state, str) else json.dumps(initial_state, ensure_ascii=False)
        called_tools: List[str] = []
        traces: List[RoundTrace] = []
        state: Optional[RexState] = None
        last_step_out: Optional[RexStepOutput] = None
        pending_obs: bool = False
        term_reason = TerminationReason.MAX_ROUNDS_REACHED

        device = self.model.tool_embeddings.device
        general_tools = [
            spec.name for spec in (tool_specs or []) if getattr(spec, "general", False)
        ]
        if not general_tools and hasattr(self.model, "tool_meta"):
            general_tools = [t for t, meta in self.model.tool_meta.items() if meta.get("general", False)]

        for round_idx in range(1, max_rounds + 1):
            # 1. Derive tool availability mask (parity with Registry.tool_mask)
            tool_mask_list = self.mask_manager.compute_tool_mask(
                self.model.tools,
                current_alive,
                tool_hyp_mapping,
                called_tools=called_tools,
                general_tools=general_tools,
                allow_unmapped=False,
            )
            tool_mask_tensor = torch.tensor([tool_mask_list], dtype=torch.bool, device=device)

            alive_mask_list = [current_alive.get(k, True) for k in hyp_keys]
            alive_mask_tensor = torch.tensor([alive_mask_list], dtype=torch.bool, device=device)

            # 2. Forward pass through RexDecisionModel
            if round_idx == 1 or state is None:
                encoded = encode_rex_input(
                    tok,
                    type_token=type_token,
                    instruction=instruction,
                    hypotheses=hypotheses,
                    state=state_t,
                    alive=current_alive,
                    tool_mask=tool_mask_list,
                )
                batch = collate_rex([encoded], pad_token_id=getattr(tok, "pad_token_id", 0) or 0)
                batch_device = {k: v.to(device) for k, v in batch.items()}

                with torch.no_grad():
                    _, state, step_out = self.model.forward(
                        input_ids=batch_device["input_ids"],
                        attention_mask=batch_device["attention_mask"],
                        marker_pos=batch_device["marker_pos"],
                        marker_mask=batch_device["marker_mask"],
                        qtype=batch_device.get("qtype"),
                        tool_mask=tool_mask_tensor,
                        hyp_alive=alive_mask_tensor,
                        tool_specs=tool_specs,
                        hyp_names=hyp_keys,
                    )
            else:
                # Subsequent round: the previous observations are already in memory (step 5)
                with torch.no_grad():
                    _, state, step_out = self.model.forward(
                        state=state,
                        tool_mask=tool_mask_tensor,
                        hyp_alive=alive_mask_tensor,
                        tool_specs=tool_specs,
                        hyp_names=hyp_keys,
                    )
                pending_obs = False

            last_step_out = step_out

            # 3. Action and tool selection check
            selected_tools = step_out.selected_tools[0] if step_out.selected_tools else []
            proposed_action = step_out.proposed_action[0] if step_out.proposed_action else "CALL_TOOL"

            if proposed_action == "STALL":
                term_reason = TerminationReason.STALL
                break

            if proposed_action == "VERDICT" or not selected_tools:
                term_reason = TerminationReason.EMPTY_TOOL_SELECTION
                break

            called_tools.extend(selected_tools)

            # 4. Dispatch selected tools
            observations = tool_dispatcher(selected_tools)
            pending_obs = True
            traces.append(
                RoundTrace(
                    round_idx=round_idx,
                    selected_tools=selected_tools,
                    observations=observations,
                    active_hypotheses=[k for k, v in current_alive.items() if v],
                )
            )

            # 5. Ingest the new observations into memory, then score hypotheses on the updated memory
            ev_encoded = encode_evidence(
                tok, format_observations(observations), round_idx=round_idx + 1, max_tokens=64
            )
            ev_ids = torch.tensor([ev_encoded["input_ids"]], dtype=torch.long, device=device)
            ev_mask = torch.tensor([ev_encoded["attention_mask"]], dtype=torch.bool, device=device)
            with torch.no_grad():
                state = self.model.ingest_evidence(
                    state, ev_ids, ev_mask, is_truncated=bool(ev_encoded["is_truncated"])
                )
                neural_scores = self.model.hyp_scorer(state.hyp_markers, state.M, state.memory_mask)

            # Hybrid PoE Evaluation (symbolic veto + neural score)
            current_alive = self.poe_engine.evaluate(
                neural_scores=neural_scores,
                hyp_keys=hyp_keys,
                observations=observations,
                rules=rules,
                theta_elim=self.theta_elim,
                prev_alive=current_alive,
            )

            # 6. Check termination criteria
            active_count = sum(1 for v in current_alive.values() if v)
            known_active_count = sum(1 for k, v in current_alive.items() if k != "H_novel" and v)

            if known_active_count == 0:
                # All cataloged hypotheses eliminated; only H_novel remains
                term_reason = TerminationReason.ALL_CATALOGED_ELIMINATED
                break

            if active_count == 1 or (stop_on_single_survivor and known_active_count == 1):
                # Exactly one cataloged survivor remains
                term_reason = TerminationReason.SINGLE_SURVIVOR
                break

            if round_idx == max_rounds:
                term_reason = TerminationReason.MAX_ROUNDS_REACHED
                break

        # Final verdict pass over the latest ingested observations (no second ingestion, L5)
        if pending_obs and traces and last_step_out is not None and state is not None:
            alive_mask_list = [current_alive.get(k, True) for k in hyp_keys]
            alive_mask_tensor = torch.tensor([alive_mask_list], dtype=torch.bool, device=device)

            with torch.no_grad():
                _, state, last_step_out = self.model.forward(
                    state=state,
                    hyp_alive=alive_mask_tensor,
                    hyp_names=hyp_keys,
                )
            pending_obs = False

        if last_step_out is None:
            raise RuntimeError("Model failed to execute forward pass during investigation.")

        probs: Dict[str, float] = {}
        for idx, key in enumerate(hyp_keys):
            if not current_alive.get(key, True):
                probs[key] = 0.0
            else:
                probs[key] = float(last_step_out.verdict_probs[0, idx].item())

        total_prob = sum(probs.values())
        if total_prob > 0:
            probs = {k: v / total_prob for k, v in probs.items()}
        else:
            probs = {k: 1.0 / len(probs) for k in probs}

        winner = max(probs.items(), key=lambda kv: kv[1])[0]
        # Mirrors Part B stop.decide: a STALL, or a spent budget without sufficient evidence, goes to a human
        # and so does any automated verdict whose risk is above the threshold (HIGH_RISK)
        final_s = float(last_step_out.s_prob.reshape(-1)[0].item())
        final_r = float(last_step_out.r_prob.reshape(-1)[0].item())
        is_escalated = (
            term_reason in (TerminationReason.ALL_CATALOGED_ELIMINATED, TerminationReason.STALL)
            or (term_reason == TerminationReason.MAX_ROUNDS_REACHED and final_s < DEFAULT_S_THRESHOLD)
            or final_r > DEFAULT_R_THRESHOLD
            or winner == "H_novel"
        )

        return InvestigationResult(
            predicted_hypothesis=winner,
            probabilities=probs,
            rounds_executed=len(traces) if traces else 1,
            termination_reason=term_reason,
            is_escalated=is_escalated,
            traces=traces,
        )
