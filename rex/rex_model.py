"""Rex unified model architecture: RexDecisionModel, 4 decision heads, and checkpoint utilities (Step A8)."""
from __future__ import annotations

import dataclasses
import warnings
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .common import QTYPES, TEMP_MAX, TEMP_MIN, DecisionModel, clamp_temperature
from .rex_heads import ConformalCalibrator, HypothesisScorer, RiskHead, SufficiencyHead, VerdictBridge
from .rex_memory import EvidenceMemoryManager
from .rex_ponder import LatentPonderBlock
from .rex_registry import ToolSpec
from .rex_tools import (
    DEFAULT_R_THRESHOLD,
    DEFAULT_S_THRESHOLD,
    ToolGainHead,
    ToolMatcherHead,
    evaluate_stall_and_dispatch,
)
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
    "alive_count_emb.",
    "tool_embeddings",
    "gain_head_trained",
    "calibrator_q_hat",
)

# Sufficiency sees how many cataloged hypotheses survive (0, 1, ..., MAX_ALIVE_COUNT or more)
MAX_ALIVE_COUNT: int = 5

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
        tool_meta: Optional[Dict[str, Dict[str, Any]]] = None,
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
        # Dispatcher metadata (level, cost) per tool, kept in the checkpoint config (rex_config)
        self.tool_meta: Dict[str, Dict[str, Any]] = {}
        self._warned_missing_tool_meta = False
        specs_given = [t for t in tools if isinstance(t, ToolSpec)]
        if specs_given:
            self.set_tool_specs(specs_given, require_all=False)
        if tool_meta:
            self.tool_meta.update({k: dict(v) for k, v in tool_meta.items() if k in self.tool_to_idx})
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

            # Survivor-count embedding added to h_T for the sufficiency head. S* is defined on the
            # survival set (only the target left among cataloged hypotheses), which the model knows
            # from the PoE mask; zero-initialized so an untrained checkpoint is unchanged.
            self.alive_count_emb = nn.Embedding(MAX_ALIVE_COUNT + 1, d)
            nn.init.zeros_(self.alive_count_emb.weight)

            # Pre-initialize tool representation buffer
            tool_init = torch.randn(len(self.tools), d) * 0.02
            self.register_buffer("tool_embeddings", tool_init)
            self.tool_matcher.set_tool_embeddings(self.tool_embeddings)

            # 6. Memory Manager & Calibrator
            self.memory_manager = EvidenceMemoryManager()
            self.calibrator = ConformalCalibrator(alpha=calibrator_alpha)
            self.register_buffer("gain_head_trained", torch.tensor(False))
            self.register_buffer("calibrator_q_hat", torch.tensor(0.9))

        if no_init:
            self.ponder_block.to_empty(device="cpu")
            self.head_s.to_empty(device="cpu")
            self.head_r.to_empty(device="cpu")
            self.verdict_bridge.to_empty(device="cpu")
            self.tool_matcher.to_empty(device="cpu")
            self.tool_gain.to_empty(device="cpu")
            self.hyp_scorer.to_empty(device="cpu")
            self.alive_count_emb.to_empty(device="cpu")
            self.tool_embeddings = torch.empty((len(self.tools), d), device="cpu")
            self.tool_matcher.set_tool_embeddings(self.tool_embeddings)
            self.gain_head_trained = torch.tensor(False)
            self.calibrator_q_hat = torch.tensor(0.9)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load weights; checkpoints saved before `alive_count_emb` existed get it zero-filled."""
        key = "alive_count_emb.weight"
        if key not in state_dict:
            state_dict = dict(state_dict)
            state_dict[key] = torch.zeros_like(self.alive_count_emb.weight, device="cpu")
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    @property
    def tool_head(self) -> ToolMatcherHead:
        """Backward-compatibility alias mapping legacy tool_head to tool_matcher."""
        return self.tool_matcher

    def encode_tool_descriptions(self, tool_specs: Sequence[ToolSpec], tok: Any) -> None:
        """Encode textual tool specifications into tool_embeddings buffer using model's encoder (T5)."""
        specs_by_name = {s.name: s for s in tool_specs}
        dev = next(self.parameters()).device
        encoded_vectors = []

        was_training = self.training
        self.eval()
        with torch.no_grad():
            for tool_name in self.tools:
                spec = specs_by_name.get(tool_name)
                desc = spec.description if (spec and getattr(spec, "description", None)) else f"diagnostic tool {tool_name}"
                inputs = tok(desc, return_tensors="pt", truncation=True, max_length=64)
                input_ids = inputs["input_ids"].to(dev)
                attn_mask = inputs["attention_mask"].to(dev)
                qtype = torch.zeros(1, dtype=torch.long, device=dev)
                h = self._contextualize(input_ids, attn_mask, qtype)
                # Use [CLS] representation (token 0)
                emb = h[0, 0].detach()
                encoded_vectors.append(emb)

        new_embeddings = torch.stack(encoded_vectors, dim=0)  # [N_tools, d]
        self.tool_embeddings.copy_(new_embeddings)
        self.tool_matcher.set_tool_embeddings(self.tool_embeddings)
        if was_training:
            self.train()

    def set_tool_specs(self, tool_specs: Sequence[ToolSpec], require_all: bool = True) -> None:
        """Store each tool's dispatcher metadata (level, cost); used when forward() gets no tool_specs."""
        given = {s.name: s for s in tool_specs if s.name in self.tool_to_idx}
        missing = [t for t in self.tools if t not in given]
        if require_all and missing:
            raise ValueError(f"tool_specs is missing model tools: {missing}")
        for name, spec in given.items():
            self.tool_meta[name] = {"level": getattr(spec, "level", "READ"), "cost": float(getattr(spec, "cost", 0.0))}

    def _resolve_tool_meta(self, tool_specs: Optional[Sequence[Any]]) -> Tuple[Dict[str, str], Dict[str, float]]:
        """(levels, costs) per model tool: explicit tool_specs, else the stored metadata."""
        if tool_specs:
            specs = [t if isinstance(t, ToolSpec) else ToolSpec(name=str(t), returns=[]) for t in tool_specs]
            by_name = {s.name: s for s in specs}
            missing = [t for t in self.tools if t not in by_name]
            if missing:
                raise ValueError(f"tool_specs is missing model tools: {missing}; their level/cost would be guessed")
            meta = {n: {"level": getattr(by_name[n], "level", "READ"), "cost": float(getattr(by_name[n], "cost", 0.0))}
                    for n in self.tools}
        else:
            missing = [t for t in self.tools if t not in self.tool_meta]
            if missing and not self._warned_missing_tool_meta:
                warnings.warn(
                    f"RexDecisionModel: no ToolSpec for tools {missing}; treating them as cheap READ tools "
                    f"(cost 0). Pass tool_specs or call set_tool_specs().",
                    RuntimeWarning,
                    stacklevel=3,
                )
                self._warned_missing_tool_meta = True
            meta = {n: self.tool_meta.get(n, {"level": "READ", "cost": 0.0}) for n in self.tools}
        return ({n: m["level"] for n, m in meta.items()}, {n: float(m["cost"]) for n, m in meta.items()})

    def mark_gain_head_trained(self, trained: bool = True) -> None:
        """Record whether the gain head was trained; the dispatcher only trusts gains when True."""
        self.gain_head_trained.fill_(bool(trained))

    def calibrate_conformal(self, probs: torch.Tensor, targets: torch.Tensor) -> float:
        """Fit the conformal threshold on held-out verdict probabilities and store it in the checkpoint buffer."""
        q_hat = self.calibrator.calibrate(probs.detach().float().cpu(), targets.detach().cpu())
        self.calibrator_q_hat.fill_(q_hat)
        return q_hat

    def verdict_temperature(self) -> torch.Tensor:
        """Temperature applied to verdict logits (the Laya `choice` slot), clamped like Laya's own."""
        return self.temperature[QTYPES["choice"]].clamp(TEMP_MIN, TEMP_MAX)

    def set_verdict_temperature(self, value: float) -> None:
        """Store a fitted verdict temperature in the checkpoint buffer."""
        self.temperature[QTYPES["choice"]] = float(value)

    def conformal_threshold(self) -> float:
        """Active conformal threshold: a freshly calibrated value, else the one stored in the checkpoint."""
        if self.calibrator.q_hat is not None:
            return float(self.calibrator.q_hat)
        return float(self.calibrator_q_hat.item())

    @staticmethod
    def _resolve_hyp_alive(
        hyp_alive: torch.Tensor,
        marker_mask: Optional[torch.Tensor],
        hyp_names: Optional[List[str]],
    ) -> torch.Tensor:
        """Normalize a survival mask: padding slots stay dead and the H_novel slot (by name) stays alive."""
        if hyp_alive.dim() == 1:
            hyp_alive = hyp_alive.unsqueeze(0)
        alive = hyp_alive.bool().clone()
        if marker_mask is not None and marker_mask.shape == alive.shape:
            alive = alive & marker_mask.bool().to(alive.device)
        if hyp_names and "H_novel" in hyp_names:
            novel_idx = hyp_names.index("H_novel")
            if novel_idx < alive.size(1):
                alive[:, novel_idx] = True
        return alive

    def ingest_evidence(
        self,
        state: RexState,
        evidence_new: torch.Tensor,
        evidence_mask: Optional[torch.Tensor] = None,
        qtype: Optional[torch.Tensor] = None,
        detach_encoder: bool = False,
        is_truncated: bool = False,
    ) -> RexState:
        """Encode a new evidence piece and append it to the memory of a copy of `state`.

        Does not ponder or dispatch, so callers can score hypotheses on the updated memory
        before the next `forward(state=...)`.
        """
        new_state = dataclasses.replace(state, called_tools=[list(c) for c in state.called_tools])
        if evidence_new.dim() < 2 or evidence_new.size(1) == 0:
            # Nothing observed (e.g. every tool failed): memory is unchanged; the encoder cannot take 0 tokens
            return new_state
        B = new_state.h.size(0)
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
        return self.memory_manager.append_evidence(
            new_state,
            h_ev,
            evidence_mask.bool(),
            is_truncated=is_truncated,
            ponder_block=self.ponder_block,
        )

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
        is_truncated: bool = False,
        dispatch: bool = True,
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
            hyp_names: Optional names of candidate hypotheses (in marker order); locates the H_novel slot.
            detach_encoder: Whether to detach encoder outputs during training.
            position_ids: Positional encoding overrides.
            option_ids: Option layout indicators.
            ponder_steps: Number of pondering steps T (default: self.ponder_steps).
            is_truncated: Whether evidence observation was truncated (N2).
            dispatch: Run conformal sets and the per-sample STALL/tool dispatcher. Training passes
                False to skip this host-side loop; decision fields are then left empty.

        Returns:
            Tuple of:
                verdict_logits: Float tensor [B, K].
                updated_state: Updated RexState carrier.
                step_output: Detailed RexStepOutput container with logits, probs, and actions.
        """
        if ponder_steps is not None:
            T = ponder_steps
        elif self.training:
            T = None  # LatentPonderBlock randomly samples U(2, 4) during training (L10)
        else:
            T = self.ponder_steps

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
            active_hyp = self._resolve_hyp_alive(
                hyp_alive if hyp_alive is not None else torch.ones((B, K), dtype=torch.bool, device=h.device),
                marker_mask,
                hyp_names,
            )

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
                is_truncated=is_truncated,
                marker_mask=marker_mask.bool(),
            )
        else:
            # Defensive copy avoiding in-place mutation of caller state (L11)
            current_state = dataclasses.replace(
                state,
                called_tools=[list(c) for c in state.called_tools],
            )
            B = current_state.h.size(0)

            # Propagate updated survival mask across subsequent rounds (L1)
            if hyp_alive is not None:
                current_state.hyp_alive = self._resolve_hyp_alive(hyp_alive, current_state.marker_mask, hyp_names)

            if evidence_new is not None:
                current_state = self.ingest_evidence(
                    current_state,
                    evidence_new,
                    evidence_mask,
                    qtype=qtype,
                    detach_encoder=detach_encoder,
                    is_truncated=is_truncated,
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
        if marker_mask is None:
            marker_mask = current_state.marker_mask
        effective_mask = (
            marker_mask.bool() if marker_mask is not None else torch.ones_like(raw_verdict_logits, dtype=torch.bool)
        )
        if current_state.hyp_alive is not None:
            effective_mask = effective_mask & current_state.hyp_alive
        verdict_logits = raw_verdict_logits.masked_fill(~effective_mask, -1e4)
        # Calibrated probabilities use the checkpoint's choice temperature; the raw logits feed the losses
        verdict_probs = torch.softmax(verdict_logits / self.verdict_temperature(), dim=-1)

        # 6. Sufficiency and Risk Heads
        # Cataloged survivors: live slots minus the H_novel residual, which is always alive
        n_known_alive = (effective_mask.sum(dim=-1) - 1).clamp(0, MAX_ALIVE_COUNT)
        s_logit = self.head_s(h_T + self.alive_count_emb(n_known_alive))
        s_prob = torch.sigmoid(s_logit)
        # Epistemic guardrail against survivor-count shortcut (L1):
        # When only 1 cataloged hypothesis is alive, PoE may have mistakenly pruned the true target.
        # If the surviving hypothesis does not match the evidence (high entropy / doubt between survivor and H_novel),
        # epistemic sufficiency S must be tempered to prevent premature convergence.
        is_single_survivor = (n_known_alive == 1).unsqueeze(-1) if n_known_alive.dim() == 1 else (n_known_alive == 1)
        if is_single_survivor.any():
            cat_mask = effective_mask.clone()
            cat_mask[..., -1] = False
            p_cat = (verdict_probs.detach() * cat_mask.float()).sum(dim=-1, keepdim=True)
            survivor_fit = torch.max(p_cat, verdict_probs.detach()[..., -1:])
            s_prob = torch.where(is_single_survivor, torch.min(s_prob, (survivor_fit * 1.25).clamp(0.0, 1.0)), s_prob)
            s_logit = torch.logit(s_prob.clamp(1e-6, 1.0 - 1e-6))
        # Risk factorizes over hypotheses: R = sum_j p_j * r_j, where r_j is the business risk of
        # hypothesis j being the true cause (read from its bridged marker) and p_j the verdict
        # probability. p_j is detached so the risk loss does not pull the verdict.
        hyp_risk_logits = self.head_r(m_bridged).squeeze(-1).float()  # [B, K]
        hyp_risk = torch.sigmoid(hyp_risk_logits)
        r_prob = (verdict_probs.detach() * hyp_risk).sum(dim=-1, keepdim=True).clamp(1e-6, 1.0 - 1e-6)
        r_logit = torch.logit(r_prob)

        # 7. Tool Matcher and Gain Heads
        tool_logits = self.tool_matcher(h_T)
        if tool_mask is not None:
            tool_logits = tool_logits.masked_fill(~tool_mask, -1e4)
        tool_gain_logits = self.tool_gain(h_T, self.tool_embeddings, return_logits=True)
        tool_gains = torch.sigmoid(tool_gain_logits)

        # 8. Hypothesis survival scores over the current evidence memory (PoE neural signal)
        hyp_logits = self.hyp_scorer(
            current_state.hyp_markers, current_state.M, current_state.memory_mask, return_logits=True
        )
        hyp_scores = torch.sigmoid(hyp_logits)

        k_names = hyp_names or [f"H_{i}" for i in range(verdict_logits.size(1))]
        # Order tool_names strictly by self.tools to align with tool_logits / tool_gains columns (L7)
        tool_names = list(self.tools)
        is_gain_trained = bool(self.gain_head_trained.item())

        actions: List[str] = []
        selected_batch: List[List[str]] = [[] for _ in range(B)]
        stalls: List[bool] = [False] * B
        stall_modes: List[str] = []
        stall_reasons: List[str] = []
        rule_stalls: List[bool] = [False] * B
        conformal_sets: Optional[List[List[str]]] = None
        alive_names: List[List[str]] = []

        if dispatch:
            # 9. Conformal Calibration
            conformal_sets = self.calibrator.predict_set(
                verdict_probs, labels=k_names, q_hat=self.conformal_threshold()
            )

            # 10. STALL Evaluation & 3-Branch Dispatch
            tool_levels, tool_costs = self._resolve_tool_meta(tool_specs)

            if isinstance(current_state.last_read_tried, list):
                last_read_list = list(current_state.last_read_tried)
                if len(last_read_list) < B:
                    last_read_list.extend([False] * (B - len(last_read_list)))
            else:
                last_read_list = [bool(current_state.last_read_tried)] * B

            s_vals = s_prob.detach().float().view(-1).tolist()
            r_vals = r_prob.detach().float().view(-1).tolist()
            s_prev_list = current_state.s_prev
            alive_rows = effective_mask.tolist()
            alive_names = [
                [k_names[j] for j, ok in enumerate(row) if ok and j < len(k_names)] for row in alive_rows
            ]

            for b in range(B):
                b_conformal = conformal_sets[b] if conformal_sets else None
                b_called = current_state.called_tools[b] if b < len(current_state.called_tools) else []
                b_mask = tool_mask[b] if tool_mask is not None else None
                b_s_prev = s_prev_list[b] if s_prev_list is not None and b < len(s_prev_list) else None

                action, selected, stall, stall_mode, rule_stall, new_last_read, _ = evaluate_stall_and_dispatch(
                    tool_names=tool_names,
                    tool_logits=tool_logits[b],
                    tool_gains=tool_gains[b],
                    s_prob=s_prob[b],
                    r_prob=r_prob[b],
                    round_idx=current_state.round_idx,
                    called_tools=b_called,
                    conformal_set=b_conformal,
                    last_read_tried=last_read_list[b],
                    tool_levels=tool_levels,
                    tool_costs=tool_costs,
                    tool_mask=b_mask,
                    gain_head_trained=is_gain_trained,
                    s_prev=b_s_prev,
                )
                actions.append(action)
                selected_batch[b] = selected
                stalls[b] = stall
                stall_modes.append(stall_mode)
                rule_stalls[b] = rule_stall
                if action == "STALL":
                    # L8: sufficient-but-risky is a different situation from "no tool can help"
                    high_risk = s_vals[b] >= DEFAULT_S_THRESHOLD and r_vals[b] > DEFAULT_R_THRESHOLD
                    stall_reasons.append("HIGH_RISK" if high_risk else "NO_GAIN")
                else:
                    stall_reasons.append("NONE")
                last_read_list[b] = new_last_read
                if b < len(current_state.called_tools):
                    current_state.called_tools[b].extend(selected)

            current_state.last_read_tried = last_read_list[0] if B == 1 else last_read_list
            current_state.s_prev = s_vals

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
            stall_modes=stall_modes,
            rule_stall=torch.tensor(rule_stalls, dtype=torch.bool, device=h_T.device),
            selected_tools=selected_batch,
            proposed_action=actions,
            h_T=h_T,
            tool_names=tool_names,
            hyp_names=k_names,
            conformal_set=conformal_sets,
            attn_weights=attn_weights,
            gain_head_trained=is_gain_trained,
            is_truncated=current_state.is_truncated,
            tool_gain_logits=tool_gain_logits,
            hyp_logits=hyp_logits,
            hyp_scores=hyp_scores,
            hyp_risk=hyp_risk,
            hyp_risk_logits=hyp_risk_logits,
            stall_reason=stall_reasons,
            alive_hypotheses=alive_names,
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
        """Lightweight compatibility shim for legacy Rex v1 training and evaluation scripts.

        Scores hypotheses from the [CLS] vector without the ponder block, the evidence memory or the
        tool/sufficiency heads, so it is not the inference path: use forward() / InvestigatorAgent.
        """
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
    # build_model sizes the Laya head and act_head from these keys; keep them consistent with the model
    cfg.setdefault("head_layers", len(model.head.layers) if model.head is not None else 0)
    if "act_costs" not in cfg:
        cfg["act_costs"] = {f"act_{i}": 0 for i in range(model.act_head[-1].out_features - 1)}
    # Laya serves choice probabilities with cfg["temperature"]; keep it equal to the Rex verdict temperature
    temps = list(cfg.get("temperature", [1.0] * len(QTYPES)))
    temps[QTYPES["choice"]] = round(float(model.verdict_temperature()), 6)
    cfg["temperature"] = temps
    cfg["rex"] = {
        "tools": list(model.tools),
        "type_tokens": dict(model.type_tokens),
        "tool_hidden": model.tool_hidden,
        "tool_dropout": model.tool_dropout,
        "tool_prior": model.tool_prior,
        "tool_meta": {k: dict(v) for k, v in model.tool_meta.items()},
    }
    return cfg


def from_rex(
    laya_model: DecisionModel,
    tools: Sequence[str],
    type_tokens: Optional[Dict[str, str]] = None,
    tool_hidden: Optional[int] = None,
    tool_dropout: float = 0.1,
    tool_prior: Optional[float] = None,
    base_cfg: Optional[Dict[str, Any]] = None,
) -> RexDecisionModel:
    """Wrap an existing pre-trained Laya DecisionModel into a RexDecisionModel.

    Transfers all trained weights from the Laya model (encoder, transformer head,
    scorer, act_head, type_emb, temperature) and initializes Rex2 modules. When the Laya
    config `base_cfg` is given, its fitted choice temperature becomes the verdict temperature,
    so Rex starts with the probabilities Laya serves.
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

    if base_cfg is not None and base_cfg.get("temperature"):
        rex_model.set_verdict_temperature(clamp_temperature(base_cfg["temperature"][QTYPES["choice"]]))

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
