"""Rex unified model architecture: RexDecisionModel, 4 decision heads, and checkpoint utilities (Step A8)."""
from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import DecisionModel
from .rex_heads import ConformalCalibrator, HypothesisScorer, RiskHead, SufficiencyHead, VerdictBridge
from .rex_memory import EvidenceMemoryManager
from .rex_ponder import LatentPonderBlock
from .rex_registry import ToolSpec
from .rex_tools import ToolGainHead, ToolMatcherHead, evaluate_stall_and_dispatch
from .rex_types import RexState, RexStepOutput

# Checkpoint loader prefix whitelist for newly initialized Rex2 modules
REX2_NEW_MODULE_PREFIXES: Tuple[str, ...] = (
    "ponder_block.",
    "head_s.",
    "head_r.",
    "verdict_bridge.",
    "tool_matcher.",
    "tool_gain.",
    "hyp_scorer.",
    "tool_embeddings",
)

# Deprecated alias for legacy v1 imports
ToolHead = ToolMatcherHead


class RexDecisionModel(DecisionModel):
    """Rex unified decision model (arch='rex2') extending Laya's DecisionModel.

    Architecture:
        - Latent Ponder Block with Cross-Attention over evidence memory and KV caching
        - 4 Decision Heads: SufficiencyHead (S), RiskHead (R), ToolGainHead, ToolMatcherHead
        - Zero-init Verdict Bridge for hypothesis logits with exact Laya checkpoint parity
        - HypothesisScorer for PoE likelihood estimation
        - Unified step-by-step forward execution with RexState and RexStepOutput
    """

    def __init__(
        self,
        encoder: nn.Module,
        tools: Sequence[Union[str, ToolSpec]],
        type_tokens: Optional[Dict[str, str]] = None,
        tool_hidden: Optional[int] = None,
        tool_dropout: float = 0.1,
        tool_prior: Optional[float] = None,
        head_layers: int = 2,
        n_act: int = 2,
        dropout: float = 0.1,
        no_init: bool = False,
        ponder_steps: int = 3,
        calibrator_alpha: float = 0.10,
    ):
        super().__init__(encoder, head_layers=head_layers, n_act=n_act, dropout=dropout, no_init=no_init)
        if not tools:
            raise ValueError("tools list must not be empty")

        cleaned_tools: List[str] = []
        for t in tools:
            name = t.name if isinstance(t, ToolSpec) else t
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"Tool names must be non-empty strings, got {name!r}")
            cleaned_tools.append(name.strip())

        if len(cleaned_tools) != len(set(cleaned_tools)):
            seen = set()
            duplicates = [t for t in cleaned_tools if t in seen or seen.add(t)]
            raise ValueError(f"Duplicate tool names found in tools list: {duplicates}")

        self.tools = cleaned_tools
        self.tool_to_idx = {name: i for i, name in enumerate(self.tools)}
        self.type_tokens = dict(type_tokens or {})
        self.tool_hidden = tool_hidden
        self.tool_dropout = tool_dropout
        self.tool_prior = tool_prior
        self.ponder_steps = ponder_steps

        d = encoder.config.hidden_size
        # Find head count dividing d_model cleanly
        n_heads = 1
        for h in range(min(8, d), 0, -1):
            if d % h == 0:
                n_heads = h
                break

        with torch.device("meta") if no_init else nullcontext():
            # 1. Latent Ponder Block with KV-caching
            self.ponder_block = LatentPonderBlock(d_model=d, nhead=n_heads, dropout=dropout)

            # 2. Sufficiency & Risk Heads
            self.head_s = SufficiencyHead(d=d)
            self.head_r = RiskHead(d=d)

            # 3. Verdict Bridge (zero initialized for exact Laya parity)
            self.verdict_bridge = VerdictBridge(d=d)

            # 4. Tool Matcher and Gain Heads
            proj_dim = min(256, d)
            self.tool_matcher = ToolMatcherHead(d=d, proj_dim=proj_dim)
            self.tool_gain = ToolGainHead(d=d, hidden_dim=proj_dim)

            # 5. Hypothesis Scorer for PoE
            self.hyp_scorer = HypothesisScorer(d=d, nhead=n_heads)

            # Pre-initialize tool representation buffer
            tool_init = torch.randn(len(self.tools), d) * 0.02
            self.register_buffer("tool_embeddings", tool_init)
            self.tool_matcher.set_tool_embeddings(self.tool_embeddings)

            # 6. Memory Manager & Calibrator
            self.memory_manager = EvidenceMemoryManager()
            self.calibrator = ConformalCalibrator(alpha=calibrator_alpha)

        if no_init:
            self.ponder_block.to_empty(device="cpu")
            self.head_s.to_empty(device="cpu")
            self.head_r.to_empty(device="cpu")
            self.verdict_bridge.to_empty(device="cpu")
            self.tool_matcher.to_empty(device="cpu")
            self.tool_gain.to_empty(device="cpu")
            self.hyp_scorer.to_empty(device="cpu")
            self.tool_embeddings = torch.empty((len(self.tools), d), device="cpu")
            self.tool_matcher.set_tool_embeddings(self.tool_embeddings)

    @property
    def tool_head(self) -> ToolMatcherHead:
        """Backward-compatibility alias mapping legacy tool_head to tool_matcher."""
        return self.tool_matcher

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        marker_pos: Optional[torch.Tensor] = None,
        marker_mask: Optional[torch.Tensor] = None,
        qtype: Optional[torch.Tensor] = None,
        state: Optional[RexState] = None,
        evidence_new: Optional[torch.Tensor] = None,
        evidence_mask: Optional[torch.Tensor] = None,
        tool_specs: Optional[Sequence[Any]] = None,
        tool_mask: Optional[torch.Tensor] = None,
        hyp_alive: Optional[torch.Tensor] = None,
        hyp_names: Optional[List[str]] = None,
        detach_encoder: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        option_ids: Optional[torch.Tensor] = None,
        ponder_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, RexState, RexStepOutput]:
        """Unified step-by-step forward execution across multi-round investigation turns.

        Args:
            input_ids: Token IDs for initial contextualization [B, S].
            attention_mask: Attention mask for input tokens [B, S].
            marker_pos: Marker token positions for hypotheses [B, K].
            marker_mask: Boolean mask for valid hypothesis markers [B, K].
            qtype: Question type tensor [B].
            state: RexState from previous turn (None on turn 1).
            evidence_new: New observation tokens or embeddings [B, L_new].
            evidence_mask: Mask for new observation tokens [B, L_new].
            tool_specs: Catalogue of tool specifications (ToolSpec).
            tool_mask: Boolean mask [B, N_tools] disabling unavailable tools.
            hyp_alive: Boolean survival mask [B, K] across Popperian candidate hypotheses.
            hyp_names: Optional names of candidate hypotheses.
            detach_encoder: Whether to detach encoder outputs during training.
            position_ids: Positional encoding overrides.
            option_ids: Option layout indicators.
            ponder_steps: Number of pondering steps T (default: self.ponder_steps).

        Returns:
            Tuple of:
                verdict_logits: Float tensor [B, K].
                updated_state: Updated RexState carrier.
                step_output: Detailed RexStepOutput container with logits, probs, and actions.
        """
        T = ponder_steps if ponder_steps is not None else self.ponder_steps

        if state is None:
            if input_ids is None or attention_mask is None or marker_pos is None or marker_mask is None:
                raise ValueError("Turn 1 requires input_ids, attention_mask, marker_pos, and marker_mask.")

            if qtype is None:
                qtype = torch.zeros(input_ids.size(0), dtype=torch.long, device=input_ids.device)

            # 1. Contextualize initial problem statement & hypothesis tokens
            h = self._contextualize(
                input_ids,
                attention_mask,
                qtype,
                detach_encoder=detach_encoder,
                position_ids=position_ids,
                option_ids=option_ids,
            )
            B, S, d = h.shape

            # 2. Extract hypothesis markers m_j from marker positions
            idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, d)
            m = torch.gather(h, 1, idx)  # [B, K, d]

            # 3. Initial thought state h_0 and initial evidence memory M_0
            h_0 = h[:, 0]  # [CLS] token [B, d]
            M_0 = h        # [B, S, d]
            mem_mask = attention_mask.bool()  # [B, S]

            # 4. Latent Pondering
            h_T, kv_cache, attn_weights = self.ponder_block.ponder(
                h_0, M_0, memory_mask=mem_mask, T=T, kv_cache=None
            )

            # Initialize survival mask if not provided
            K = marker_pos.size(1)
            active_hyp = hyp_alive if hyp_alive is not None else torch.ones((B, K), dtype=torch.bool, device=h.device)

            current_state = RexState(
                h=h_T,
                M=M_0,
                memory_mask=mem_mask,
                kv_cache=kv_cache,
                hyp_alive=active_hyp,
                hyp_markers=m,
                round_idx=1,
                called_tools=[[] for _ in range(B)],
                last_read_tried=False,
                is_truncated=False,
            )
        else:
            current_state = state
            B = current_state.h.size(0)
            d = current_state.h.size(1)

            if evidence_new is not None:
                # Encode new evidence tokens
                if evidence_mask is None:
                    evidence_mask = torch.ones_like(evidence_new, dtype=torch.bool)
                if qtype is None:
                    qtype = torch.zeros(B, dtype=torch.long, device=evidence_new.device)

                h_ev = self._contextualize(
                    evidence_new,
                    evidence_mask,
                    qtype,
                    detach_encoder=detach_encoder,
                )
                current_state = self.memory_manager.append_evidence(
                    current_state,
                    h_ev,
                    evidence_mask,
                    is_truncated=False,
                    ponder_block=self.ponder_block,
                )

            # Ponder with updated evidence memory & existing KV-cache
            h_T, kv_cache, attn_weights = self.ponder_block.ponder(
                current_state.h,
                current_state.M,
                memory_mask=current_state.memory_mask,
                T=T,
                kv_cache=current_state.kv_cache,
            )
            current_state.h = h_T
            current_state.kv_cache = kv_cache
            current_state.round_idx += 1

        # 5. Verdict Bridge: bridge markers with h_T
        m_bridged = self.verdict_bridge(current_state.hyp_markers, h_T)
        raw_verdict_logits = self.scorer(m_bridged).squeeze(-1).float()  # [B, K]

        # Apply marker mask and PoE survival mask
        effective_mask = marker_mask if marker_mask is not None else torch.ones_like(raw_verdict_logits, dtype=torch.bool)
        if current_state.hyp_alive is not None:
            effective_mask = effective_mask & current_state.hyp_alive
        verdict_logits = raw_verdict_logits.masked_fill(~effective_mask, -1e4)
        verdict_probs = torch.softmax(verdict_logits, dim=-1)

        # 6. Sufficiency and Risk Heads
        s_logit = self.head_s(h_T)
        s_prob = torch.sigmoid(s_logit)
        r_logit = self.head_r(h_T)
        r_prob = torch.sigmoid(r_logit)

        # 7. Tool Matcher and Gain Heads
        tool_logits = self.tool_matcher(h_T)
        if tool_mask is not None:
            tool_logits = tool_logits.masked_fill(~tool_mask, -1e4)
        tool_gains = self.tool_gain(h_T, self.tool_embeddings)

        # 8. Conformal Calibration
        k_names = hyp_names or [f"H_{i}" for i in range(verdict_logits.size(1))]
        conformal_sets = self.calibrator.predict_set(verdict_probs, labels=k_names)

        # 9. STALL Evaluation & 3-Branch Dispatch
        resolved_tools = [
            t if isinstance(t, ToolSpec) else ToolSpec(name=str(t), returns=[])
            for t in (tool_specs or self.tools)
        ]
        tool_names = [t.name for t in resolved_tools]
        tool_levels = {t.name: getattr(t, "level", "READ") for t in resolved_tools}
        tool_costs = {t.name: getattr(t, "cost", 0.0) for t in resolved_tools}

        actions: List[str] = []
        selected_batch: List[List[str]] = []
        stalls: List[bool] = []
        stall_modes: List[str] = []
        rule_stalls: List[bool] = []
        last_read_state = current_state.last_read_tried

        for b in range(B):
            b_conformal = conformal_sets[b] if conformal_sets else None
            b_called = current_state.called_tools[b] if b < len(current_state.called_tools) else []
            b_mask = tool_mask[b] if tool_mask is not None else None

            action, selected, stall, stall_mode, rule_stall, new_last_read, _ = evaluate_stall_and_dispatch(
                tool_names=tool_names,
                tool_logits=tool_logits[b],
                tool_gains=tool_gains[b],
                s_prob=s_prob[b],
                r_prob=r_prob[b],
                round_idx=current_state.round_idx,
                called_tools=b_called,
                conformal_set=b_conformal,
                last_read_tried=last_read_state,
                tool_levels=tool_levels,
                tool_costs=tool_costs,
                tool_mask=b_mask,
            )
            actions.append(action)
            selected_batch.append(selected)
            stalls.append(stall)
            stall_modes.append(stall_mode)
            rule_stalls.append(rule_stall)
            last_read_state = new_last_read
            if b < len(current_state.called_tools):
                current_state.called_tools[b].extend(selected)

        current_state.last_read_tried = last_read_state

        step_output = RexStepOutput(
            s_logit=s_logit,
            s_prob=s_prob,
            r_logit=r_logit,
            r_prob=r_prob,
            verdict_logits=verdict_logits,
            verdict_probs=verdict_probs,
            tool_logits=tool_logits,
            tool_gains=tool_gains,
            stall=torch.tensor(stalls, dtype=torch.bool, device=h_T.device),
            stall_mode=stall_modes[0] if stall_modes else "NONE",
            rule_stall=torch.tensor(rule_stalls, dtype=torch.bool, device=h_T.device),
            selected_tools=selected_batch,
            proposed_action=actions,
            h_T=h_T,
            tool_names=[t.name for t in resolved_tools],
            hyp_names=k_names,
            conformal_set=conformal_sets,
            attn_weights=attn_weights,
            gain_head_trained=True,
            is_truncated=current_state.is_truncated,
        )

        return verdict_logits, current_state, step_output

    def step(self, *args, **kwargs) -> Tuple[torch.Tensor, RexState, RexStepOutput]:
        """Convenience alias for forward()."""
        return self.forward(*args, **kwargs)

    def rex_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        marker_pos: torch.Tensor,
        marker_mask: torch.Tensor,
        qtype: Optional[torch.Tensor] = None,
        hyp_alive: Optional[torch.Tensor] = None,
        tool_mask: Optional[torch.Tensor] = None,
        detach_encoder: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        option_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Lightweight compatibility shim for legacy Rex v1 training and evaluation scripts."""
        if qtype is None:
            qtype = torch.zeros(input_ids.size(0), dtype=torch.long, device=input_ids.device)

        h = self._contextualize(
            input_ids,
            attention_mask,
            qtype,
            detach_encoder=detach_encoder,
            position_ids=position_ids,
            option_ids=option_ids,
        )

        h_cls = h[:, 0]
        tool_logits = self.tool_matcher(h_cls)
        if tool_mask is not None:
            tool_logits = tool_logits.masked_fill(~tool_mask, -1e4)

        effective_marker_mask = marker_mask
        if hyp_alive is not None:
            effective_marker_mask = marker_mask & hyp_alive
            if not effective_marker_mask.any(dim=-1).all():
                raise ValueError("All hypotheses have been eliminated for at least one sample in the batch.")

        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        m_bridged = self.verdict_bridge(m, h_cls)
        verdict_logits = self.scorer(m_bridged).squeeze(-1).float()
        verdict_logits = verdict_logits.masked_fill(~effective_marker_mask, -1e4)

        _, act_logits = self._decide(h, attention_mask, marker_pos, effective_marker_mask)

        return {
            "verdict_logits": verdict_logits,
            "tool_logits": tool_logits,
            "act_logits": act_logits,
            "h_cls": h_cls,
        }

    def compute_verdict_probabilities(
        self,
        verdict_logits: torch.Tensor,
        hyp_alive: Optional[Dict[str, bool]] = None,
        hyp_keys: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Compute normalized softmax probabilities over surviving hypotheses."""
        if verdict_logits.dim() == 2:
            verdict_logits = verdict_logits.squeeze(0)
        k = verdict_logits.size(0)
        keys = hyp_keys or [f"H_{i}" for i in range(k)]
        masked_logits = verdict_logits.clone()
        if hyp_alive is not None:
            for idx, key in enumerate(keys):
                if not hyp_alive.get(key, True):
                    masked_logits[idx] = -1e4
        probs = torch.softmax(masked_logits, dim=-1)
        res = {key: float(probs[idx].item()) for idx, key in enumerate(keys)}
        if hyp_alive is not None:
            for key in keys:
                if not hyp_alive.get(key, True):
                    res[key] = 0.0
        s = sum(res.values())
        if s > 0:
            res = {k: v / s for k, v in res.items()}
        return res

    def investigate(self, *args, **kwargs) -> Any:
        """Deprecated procedural investigation method delegated to InvestigatorAgent (Step A10)."""
        from .rex_agent import InvestigatorAgent

        agent = InvestigatorAgent(model=self)
        return agent.investigate(*args, **kwargs)


def rex_config(model: RexDecisionModel, base_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Generate Rex configuration dictionary compatible with rl_agent_config.json."""
    cfg = dict(base_cfg or {})
    cfg["arch"] = "rex2"
    cfg["rex"] = {
        "tools": list(model.tools),
        "type_tokens": dict(model.type_tokens),
        "tool_hidden": model.tool_hidden,
        "tool_dropout": model.tool_dropout,
        "tool_prior": model.tool_prior,
    }
    return cfg


def from_rex(
    laya_model: DecisionModel,
    tools: Sequence[str],
    type_tokens: Optional[Dict[str, str]] = None,
    tool_hidden: Optional[int] = None,
    tool_dropout: float = 0.1,
    tool_prior: Optional[float] = None,
) -> RexDecisionModel:
    """Wrap an existing pre-trained Laya DecisionModel into a RexDecisionModel.

    Transfers all trained weights from the Laya model (encoder, transformer head,
    scorer, act_head, type_emb, temperature) and initializes Rex2 modules.
    """
    rex_model = RexDecisionModel(
        encoder=laya_model.encoder,
        tools=tools,
        type_tokens=type_tokens,
        tool_hidden=tool_hidden,
        tool_dropout=tool_dropout,
        tool_prior=tool_prior,
        head_layers=len(laya_model.head.layers) if laya_model.head is not None else 0,
        n_act=laya_model.act_head[-1].out_features,
        dropout=0.1,
        no_init=False,
    )

    rex_state = laya_model.state_dict()
    missing_keys, unexpected_keys = rex_model.load_state_dict(rex_state, strict=False)

    non_rex2_missing = [
        k for k in missing_keys
        if not any(k.startswith(p) for p in REX2_NEW_MODULE_PREFIXES)
    ]
    if non_rex2_missing:
        raise RuntimeError(f"Unexpected missing keys when loading Laya checkpoint: {non_rex2_missing}")
    if unexpected_keys:
        raise RuntimeError(f"Unexpected extra keys when loading Laya checkpoint: {unexpected_keys}")

    return rex_model


def save_rex_checkpoint(
    model: RexDecisionModel,
    tok: Any,
    output_dir: str,
    base_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """Save Rex model weights, tokenizer, encoder config, and configuration to disk."""
    from .train import save_checkpoint

    cfg = rex_config(model, base_cfg=base_cfg)
    save_checkpoint(model, tok, cfg, output_dir)
