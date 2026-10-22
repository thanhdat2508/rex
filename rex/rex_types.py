"""Standardized data containers and state contracts for Rex (Part A).

This module defines:
1. RexState: Multi-round mutable latent memory, token cache, and trajectory history.
2. RexStepOutput: Comprehensive step outputs, logits, probabilities, and telemetry serializer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

MODEL_VERSION: str = "rex2-modernbert-1024"


@dataclass
class RexState:
    """State carrier maintained across investigative pondering and multi-round tool turns.

    Attributes:
        h: Contextualized thought state tensor of shape [B, d].
        M: Contextualized evidence memory tokens of shape [B, L_mem, d].
        memory_mask: Boolean validity mask of shape [B, L_mem] (True = valid, False = pad).
        kv_cache: Optional tuple of cached Key and Value tensors for ponder cross-attention.
        hyp_alive: Boolean survival mask of shape [B, K] across Popperian candidate hypotheses.
        hyp_markers: Marker embeddings [B, K, d] corresponding to [MASK] positions from turn 1.
        round_idx: Current turn index (1-indexed: 1, 2, 3...).
        called_tools: List of tool names called so far per batch sample.
        last_read_tried: Guardrail flag indicating if a last-resort cheap read was attempted.
        is_truncated: Whether evidence observation was truncated by the 64-token cap.
        marker_mask: Boolean validity mask [B, K] of hypothesis slots from turn 1 (False = padding).
        s_prev: Sufficiency probability per sample from the previous round (for rule_stall telemetry).
    """

    h: torch.Tensor
    M: torch.Tensor
    memory_mask: torch.Tensor
    kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    hyp_alive: Optional[torch.Tensor] = None
    hyp_markers: Optional[torch.Tensor] = None
    round_idx: int = 1
    called_tools: List[List[str]] = field(default_factory=list)
    last_read_tried: Union[bool, List[bool]] = False
    is_truncated: bool = False
    marker_mask: Optional[torch.Tensor] = None
    s_prev: Optional[List[float]] = None

    @classmethod
    def empty(
        cls,
        batch_size: int = 1,
        d: int = 1024,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> RexState:
        """Create an empty initialization state for turn 1 before initial encoding."""
        dev = device if device is not None else torch.device("cpu")
        return cls(
            h=torch.zeros((batch_size, d), dtype=dtype, device=dev),
            M=torch.zeros((batch_size, 0, d), dtype=dtype, device=dev),
            memory_mask=torch.zeros((batch_size, 0), dtype=torch.bool, device=dev),
            kv_cache=None,
            hyp_alive=None,
            hyp_markers=None,
            round_idx=1,
            called_tools=[[] for _ in range(batch_size)],
            last_read_tried=False,
            is_truncated=False,
        )


@dataclass
class RexStepOutput:
    """Output bundle produced by RexDecisionModel.step().

    Contains both raw unnormalized logits (for numerical loss stability) and normalized
    probabilities, calibration metrics, and selected actions.
    """

    s_logit: torch.Tensor                        # [B, 1]
    s_prob: torch.Tensor                         # [B, 1]
    r_logit: torch.Tensor                        # [B, 1] logit(r_prob)
    r_prob: torch.Tensor                         # [B, 1] sum_j verdict_probs_j * hyp_risk_j
    verdict_logits: torch.Tensor                 # [B, K]
    verdict_probs: torch.Tensor                  # [B, K]
    tool_logits: torch.Tensor                    # [B, N_tools]
    tool_gains: torch.Tensor                     # [B, N_tools]
    stall: torch.Tensor                          # [B] bool
    stall_mode: str                              # "LEARNED" or "NONE"
    rule_stall: torch.Tensor                     # [B] bool
    selected_tools: List[List[str]]              # Approved tool names per batch item
    proposed_action: List[str]                   # "CALL_TOOL", "VERDICT", or "STALL"
    h_T: torch.Tensor                            # [B, d]
    tool_names: List[str] = field(default_factory=list)
    hyp_names: List[str] = field(default_factory=list)
    conformal_set: Optional[List[List[str]]] = None
    attn_weights: Optional[torch.Tensor] = None
    gain_head_trained: bool = False
    is_truncated: bool = False
    tool_gain_logits: Optional[torch.Tensor] = None     # [B, N_tools] raw logits of tool_gains (for BCEWithLogits)
    hyp_logits: Optional[torch.Tensor] = None           # [B, K] HypothesisScorer survival logits over memory M
    hyp_scores: Optional[torch.Tensor] = None           # [B, K] sigmoid(hyp_logits)
    hyp_risk: Optional[torch.Tensor] = None             # [B, K] sigmoid(hyp_risk_logits)
    hyp_risk_logits: Optional[torch.Tensor] = None      # [B, K] logits of hyp_risk
    stall_modes: Optional[List[str]] = None
    stall_reason: List[str] = field(default_factory=list)          # per sample: "NONE", "NO_GAIN" or "HIGH_RISK"
    alive_hypotheses: List[List[str]] = field(default_factory=list)  # surviving hypothesis names per sample

    def to_trace_dict(self, batch_idx: int = 0) -> Dict[str, Any]:
        """Serialize output for trace logging and State Store.

        Strict contract: Returns purely primitive Python types (float, int, str, bool, list, dict)
        with zero torch.Tensor instances to guarantee crash-free JSON/logging serialization.
        """
        # 1. Probabilities
        s_val = float(self.s_prob[batch_idx].item()) if self.s_prob.dim() > 1 else float(self.s_prob.item())
        r_val = float(self.r_prob[batch_idx].item()) if self.r_prob.dim() > 1 else float(self.r_prob.item())

        # 2. Tool gains dictionary
        tool_gains_dict: Dict[str, float] = {}
        top_tools_list: List[Dict[str, Any]] = []
        n_tools = self.tool_logits.size(-1) if self.tool_logits.dim() > 1 else len(self.tool_names)

        if self.tool_names and len(self.tool_names) == n_tools:
            b_logits = self.tool_logits[batch_idx] if self.tool_logits.dim() > 1 else self.tool_logits
            b_gains = self.tool_gains[batch_idx] if self.tool_gains.dim() > 1 else self.tool_gains
            for idx, name in enumerate(self.tool_names):
                tool_gains_dict[name] = round(float(b_gains[idx].item()), 4)

            # Top 3 tools by relevance score
            sorted_indices = torch.argsort(b_logits, descending=True).tolist()
            for idx in sorted_indices[:3]:
                top_tools_list.append({
                    "name": self.tool_names[idx],
                    "score": round(float(torch.sigmoid(b_logits[idx]).item()), 4),
                    "gain": tool_gains_dict[self.tool_names[idx]],
                })

        # 3. Verdict probabilities dictionary
        verdict_probs_dict: Dict[str, float] = {}
        b_vprobs = self.verdict_probs[batch_idx] if self.verdict_probs.dim() > 1 else self.verdict_probs
        if self.hyp_names and len(self.hyp_names) == b_vprobs.size(-1):
            for idx, name in enumerate(self.hyp_names):
                verdict_probs_dict[name] = round(float(b_vprobs[idx].item()), 4)
        else:
            for idx in range(b_vprobs.size(-1)):
                verdict_probs_dict[f"H_{idx}"] = round(float(b_vprobs[idx].item()), 4)

        # 4. Status flags
        stall_val = bool(self.stall[batch_idx].item()) if self.stall.dim() > 0 else bool(self.stall.item())
        rule_stall_val = (
            bool(self.rule_stall[batch_idx].item()) if self.rule_stall.dim() > 0 else bool(self.rule_stall.item())
        )
        conf_set = self.conformal_set[batch_idx] if self.conformal_set else []

        return {
            "s_prob": round(s_val, 4),
            "r_prob": round(r_val, 4),
            "action": self.proposed_action[batch_idx] if self.proposed_action else "CALL_TOOL",
            "selected_tools": self.selected_tools[batch_idx] if self.selected_tools else [],
            "top_tools": top_tools_list,
            "tool_gains": tool_gains_dict,
            "verdict_probs": verdict_probs_dict,
            "conformal_set": conf_set,
            "stall": stall_val,
            "stall_mode": self.stall_modes[batch_idx] if (self.stall_modes and batch_idx < len(self.stall_modes)) else self.stall_mode,
            "stall_reason": self.stall_reason[batch_idx] if self.stall_reason else "NONE",
            "rule_stall": rule_stall_val,
            "alive_hypotheses": list(self.alive_hypotheses[batch_idx]) if self.alive_hypotheses else [],
            "gain_head_trained": self.gain_head_trained,
            "is_truncated": self.is_truncated,
            "model_version": MODEL_VERSION,
        }
