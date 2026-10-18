"""Rex model architecture: RexDecisionModel, ToolHead, and checkpoint utilities."""
import math
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn

from .common import DecisionModel

# Default hyperparameters for Dynamic Margin Gating and Budget Allocation
DEFAULT_S_MIN: float = 0.35
DEFAULT_DELTA_MARGIN: float = 0.15
DEFAULT_BUDGET_CAPS: tuple[int, int, int] = (3, 2, 1)


class ToolHead(nn.Module):
    """Tool selection MLP head for Rex.

    Architecture:
        h[:, 0] ([CLS]) -> Linear(d, hidden_dim) -> LayerNorm -> GELU -> Dropout -> Linear(hidden_dim, N_tools)

    Notes:
        - LayerNorm in the hidden layer mitigates anisotropic representation drift.
        - Output bias is initialized to logit(prior) for numerical stability under focal loss.
        - Raw float32 logits are returned; thresholding / margin / Top-K selection lives in runtime.
    """

    def __init__(
        self,
        d: int,
        n_tools: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        tool_prior: Optional[float] = None,
    ):
        super().__init__()
        hidden_dim = hidden_dim or d
        self.d = d
        self.n_tools = n_tools
        self.hidden_dim = hidden_dim

        self.proj_in = nn.Linear(d, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Linear(hidden_dim, n_tools)

        # Initialize weights
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.zeros_(self.proj_in.bias)
        nn.init.xavier_uniform_(self.proj_out.weight)

        # Output bias initialized to logit(prior) to handle extreme label sparsity
        if n_tools > 0:
            if tool_prior is not None:
                prior = tool_prior
            else:
                prior = min(max(2.0 / max(1, n_tools), 1e-4), 0.5)
            logit_prior = math.log(prior / (1.0 - prior))
            nn.init.constant_(self.proj_out.bias, logit_prior)
        else:
            nn.init.zeros_(self.proj_out.bias)

    @property
    def weight(self) -> torch.Tensor:
        """Weight matrix [n_tools, hidden_dim] for orthogonality loss computation."""
        return self.proj_out.weight

    @property
    def tool_embeddings(self) -> torch.Tensor:
        """Alias for projection weight matrix [n_tools, hidden_dim]."""
        return self.proj_out.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute tool selection logits from contextualized [CLS] embedding.

        Parameters
        ----------
        x : torch.Tensor
            Contextualized [CLS] hidden states, shape [batch_size, d].

        Returns
        -------
        torch.Tensor
            Raw logits, shape [batch_size, n_tools], dtype float32.
        """
        h = self.proj_in(x)
        h = self.ln(h)
        h = self.act(h)
        h = self.dropout(h)
        return self.proj_out(h).float()

    def select_tools(
        self,
        h_cls: torch.Tensor,
        round_idx: int,
        tool_mask: Optional[torch.Tensor] = None,
        s_min: float = DEFAULT_S_MIN,
        delta_margin: float = DEFAULT_DELTA_MARGIN,
        budget_caps: tuple[int, ...] = DEFAULT_BUDGET_CAPS,
    ) -> List[int]:
        """Select parallel tools using Sigmoid + Dynamic Margin Gating + Descending Budget Cap.

        Parameters
        ----------
        h_cls : torch.Tensor
            Hidden state of [CLS] token, shape [d] or [1, d].
        round_idx : int
            Current investigation round (1-indexed, e.g. 1, 2, 3).
        tool_mask : Optional[torch.Tensor]
            Boolean availability mask of shape [n_tools] or [1, n_tools].
            False denotes tool is disabled by PoE hard masking.
        s_min : float
            Absolute confidence floor threshold (default 0.35).
        delta_margin : float
            Relative tolerance margin below top-1 score (default 0.15).
        budget_caps : tuple[int, ...]
            Descending budget limits per round (default 3, 2, 1).

        Returns
        -------
        List[int]
            Indices of selected tools.
        """
        if h_cls.dim() == 1:
            h_cls = h_cls.unsqueeze(0)

        logits = self.forward(h_cls).squeeze(0)  # [n_tools]

        if tool_mask is not None:
            mask = tool_mask.squeeze(0) if tool_mask.dim() == 2 else tool_mask
            logits = logits.masked_fill(~mask, -1e4)

        # Layer 1: Independent Sigmoid Scoring
        scores = torch.sigmoid(logits)

        # Layer 2: Dynamic Margin Gating
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        top_1_score = sorted_scores[0].item()

        # If top tool falls below s_min floor or is masked out (-1e4 -> sigmoid ~ 0), return empty
        if top_1_score < s_min or torch.isinf(sorted_scores[0]):
            return []

        selected_indices = [sorted_indices[0].item()]

        # Determine round-dependent budget cap K_t
        cap_idx = max(0, min(round_idx - 1, len(budget_caps) - 1))
        budget_cap = budget_caps[cap_idx]

        # Scan runners-up within budget
        max_scan = min(budget_cap, len(sorted_scores))
        for i in range(1, max_scan):
            score_i = sorted_scores[i].item()
            if score_i >= s_min and (top_1_score - score_i) <= delta_margin:
                selected_indices.append(sorted_indices[i].item())
            else:
                break

        return selected_indices


class RexDecisionModel(DecisionModel):
    """Rex Decision Model extending Laya's DecisionModel with parallel tool selection and PoE.

    Inherits contextualization (encoder + head layers) and hypothesis scoring (scorer)
    from DecisionModel, and adds a dedicated ToolHead alongside act_head.
    """

    def __init__(
        self,
        encoder: nn.Module,
        tools: Sequence[str],
        type_tokens: Optional[Dict[str, str]] = None,
        tool_hidden: Optional[int] = None,
        tool_dropout: float = 0.1,
        tool_prior: Optional[float] = None,
        head_layers: int = 2,
        n_act: int = 2,
        dropout: float = 0.1,
        no_init: bool = False,
    ):
        super().__init__(encoder, head_layers=head_layers, n_act=n_act, dropout=dropout, no_init=no_init)
        if not tools:
            raise ValueError("tools list must not be empty")
        cleaned_tools = []
        for t in tools:
            if not isinstance(t, str) or not t.strip():
                raise ValueError(f"Tool names must be non-empty strings, got {t!r}")
            cleaned_tools.append(t.strip())
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

        d = encoder.config.hidden_size
        with torch.device("meta") if no_init else nullcontext():
            self.tool_head = ToolHead(
                d,
                len(self.tools),
                hidden_dim=tool_hidden,
                dropout=tool_dropout,
                tool_prior=tool_prior,
            )
        if no_init:
            self.tool_head.to_empty(device="cpu")

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
        """Forward pass for Rex multi-task decision and tool selection.

        Parameters
        ----------
        input_ids : torch.Tensor
            Batch token IDs, shape [B, S].
        attention_mask : torch.Tensor
            Attention mask, shape [B, S].
        marker_pos : torch.Tensor
            Positions of [MASK] tokens corresponding to hypotheses, shape [B, K].
        marker_mask : torch.Tensor
            Padding mask for marker positions, shape [B, K] bool.
        qtype : Optional[torch.Tensor]
            Question type (default 0 for choice), shape [B].
        hyp_alive : Optional[torch.Tensor]
            PoE survival mask, shape [B, K] bool. False indicates hypothesis is eliminated.
        tool_mask : Optional[torch.Tensor]
            Mask of available tools, shape [B, N_tools] bool. False indicates tool is disabled.
        detach_encoder : bool
            Whether to detach encoder output during training.
        position_ids : Optional[torch.Tensor]
            Explicit position IDs for parallel option layout.
        option_ids : Optional[torch.Tensor]
            Explicit option IDs for parallel option layout.

        Returns
        -------
        Dict[str, torch.Tensor]
            {
                "verdict_logits": [B, K] float32,
                "tool_logits": [B, N_tools] float32,
                "act_logits": [B, n_act] float32,
                "h_cls": [B, d] hidden state,
            }
        """
        if qtype is None:
            qtype = torch.zeros(input_ids.size(0), dtype=torch.long, device=input_ids.device)

        # 1. Contextualize tokens via encoder + type embedding + head layers
        h = self._contextualize(
            input_ids,
            attention_mask,
            qtype,
            detach_encoder=detach_encoder,
            position_ids=position_ids,
            option_ids=option_ids,
        )

        # 2. Tool Head on contextualized [CLS] embedding (h[:, 0])
        h_cls = h[:, 0]
        tool_logits = self.tool_head(h_cls)
        if tool_mask is not None:
            tool_logits = tool_logits.masked_fill(~tool_mask, -1e4)

        # 3. Verdict Head on hypothesis [MASK] representations with PoE hard masking
        effective_marker_mask = marker_mask
        if hyp_alive is not None:
            effective_marker_mask = marker_mask & hyp_alive
            # Numerical sanity: ensure at least one hypothesis remains alive per sample
            if not effective_marker_mask.any(dim=-1).all():
                raise ValueError("All hypotheses have been eliminated for at least one sample in the batch.")

        verdict_logits, act_logits = self._decide(h, attention_mask, marker_pos, effective_marker_mask)

        return {
            "verdict_logits": verdict_logits,
            "tool_logits": tool_logits,
            "act_logits": act_logits,
            "h_cls": h_cls,
        }

    def select_tools(
        self,
        h_cls: torch.Tensor,
        round_idx: int,
        tool_mask: Optional[torch.Tensor] = None,
        s_min: float = DEFAULT_S_MIN,
        delta_margin: float = DEFAULT_DELTA_MARGIN,
    ) -> List[str]:
        """Select tools by name using the internal ToolHead gating pipeline (Step 5.1).

        Parameters
        ----------
        h_cls : torch.Tensor
            Contextualized [CLS] hidden state, shape [d] or [1, d].
        round_idx : int
            Current turn index (1-indexed, e.g. 1, 2, 3).
        tool_mask : Optional[torch.Tensor]
            Optional boolean mask [n_tools] disabling refuted tool branches.
        s_min : float
            Absolute confidence floor threshold.
        delta_margin : float
            Relative runner-up tolerance margin.

        Returns
        -------
        List[str]
            List of approved tool names to execute in parallel.
        """
        indices = self.tool_head.select_tools(
            h_cls,
            round_idx=round_idx,
            tool_mask=tool_mask,
            s_min=s_min,
            delta_margin=delta_margin,
        )
        return [self.tools[idx] for idx in indices]

    def derive_tool_mask(
        self,
        current_alive: Dict[str, bool],
        tool_hyp_mapping: Dict[str, List[str]],
    ) -> torch.Tensor:
        """Derive boolean tool availability tensor [n_tools] from hypothesis survival state (Step 5.3)."""
        from .rex_poe import HardMaskingManager

        manager = HardMaskingManager()
        mask_list = manager.compute_tool_mask(self.tools, current_alive, tool_hyp_mapping)
        device = self.tool_head.proj_in.weight.device
        return torch.tensor(mask_list, dtype=torch.bool, device=device)

    def evaluate_falsification(
        self,
        observations: Dict[str, Any],
        current_alive: Dict[str, bool],
        rules: Dict[str, List[Any]],
    ) -> Dict[str, bool]:
        """Evaluate observation telemetry against Popperian falsification criteria (Step 5.2)."""
        from .rex_poe import PoEFalsificationEngine

        engine = PoEFalsificationEngine()
        return engine.evaluate(observations, current_alive, rules)

    def compute_verdict_probabilities(
        self,
        verdict_logits: torch.Tensor,
        hyp_alive: Optional[Dict[str, bool]] = None,
        hyp_keys: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Compute normalized softmax probabilities over surviving hypotheses with hard logit masking (Step 5.3).

        Parameters
        ----------
        verdict_logits : torch.Tensor
            Verdict logits of shape [K] or [1, K].
        hyp_alive : Optional[Dict[str, bool]]
            Survival status per hypothesis key. Eliminated hypotheses receive -1e4 logit masking.
        hyp_keys : Optional[List[str]]
            Hypothesis names corresponding to logit indices.

        Returns
        -------
        Dict[str, float]
            Probability distribution summing to 1.0 across active hypotheses.
        """
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
        return {key: float(probs[idx].item()) for idx, key in enumerate(keys)}

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
        s_min: float = DEFAULT_S_MIN,
        delta_margin: float = DEFAULT_DELTA_MARGIN,
    ) -> Any:
        """Execute an autonomous multi-round investigation loop directly within the model (Step 5.4).

        Parameters
        ----------
        tok : Any
            Tokenizer configured with type tokens.
        type_token : str
            Domain event identifier (e.g. '<TYPE_PROMO_ABUSE>').
        instruction : str
            Core question or incident investigation prompt.
        hypotheses : Dict[str, str]
            Hypothesis name -> physical description mapping, ending with 'H_novel'.
        rules : Dict[str, List[FalsificationRule]]
            Falsification rules per hypothesis.
        tool_hyp_mapping : Dict[str, List[str]]
            Tool name -> list of supporting hypotheses.
        tool_dispatcher : Callable[[List[str]], Dict[str, Any]]
            Callback executing selected tools and returning observation telemetry.
        initial_state : Union[str, dict, list]
            Initial incident telemetry state.
        max_rounds : int
            Maximum exploration rounds before forced termination (default 3).
        s_min : float
            Confidence floor for tool gating (default 0.35).
        delta_margin : float
            Relative runner-up margin (default 0.15).

        Returns
        -------
        InvestigationResult
            Detailed verdict, probability distribution, round traces, and escalation status.
        """
        import json
        from .rex_format import collate_rex, encode_rex_input
        from .rex_poe import InvestigationResult, RoundTrace, TerminationReason

        hyp_keys = list(hypotheses.keys())
        current_alive = {k: True for k in hyp_keys}
        state_t = initial_state if isinstance(initial_state, str) else json.dumps(initial_state, ensure_ascii=False)
        traces: List[RoundTrace] = []
        final_verdict_logits: Optional[torch.Tensor] = None
        term_reason = TerminationReason.MAX_ROUNDS_REACHED

        for round_idx in range(1, max_rounds + 1):
            # 1. Derive tool availability mask from current hypothesis survival state
            tool_mask_tensor = self.derive_tool_mask(current_alive, tool_hyp_mapping)
            tool_mask_list = tool_mask_tensor.tolist()

            # 2. Format token sequence for the current investigation round
            encoded = encode_rex_input(
                tok,
                type_token=type_token,
                instruction=instruction,
                hypotheses=hypotheses,
                state=state_t,
                alive=current_alive,
                tool_mask=tool_mask_list,
            )
            pad_id = getattr(tok, "pad_token_id", 0) or 0
            batch = collate_rex([encoded], pad_token_id=pad_id)

            # Move batch tensors to model device
            device = self.tool_head.proj_in.weight.device
            batch_on_device = {k: v.to(device) for k, v in batch.items()}

            # 3. Model forward pass
            with torch.no_grad():
                outputs = self.rex_forward(**batch_on_device)

            h_cls = outputs["h_cls"]
            final_verdict_logits = outputs["verdict_logits"]

            # 4. Tool selection via Dynamic Margin Gating
            selected_tools = self.select_tools(
                h_cls,
                round_idx=round_idx,
                tool_mask=tool_mask_tensor,
                s_min=s_min,
                delta_margin=delta_margin,
            )

            # Check if tool list is empty (short-circuit to verdict)
            if not selected_tools:
                term_reason = TerminationReason.EMPTY_TOOL_SELECTION
                break

            # 5. Dispatch selected tools to retrieve observation telemetry
            observations = tool_dispatcher(selected_tools)
            traces.append(
                RoundTrace(
                    round_idx=round_idx,
                    selected_tools=selected_tools,
                    observations=observations,
                    active_hypotheses=[k for k, v in current_alive.items() if v],
                )
            )

            # 6. Apply Popperian falsification to eliminate refuted hypotheses
            current_alive = self.evaluate_falsification(observations, current_alive, rules)

            # 7. Check convergence criteria
            active_count = sum(1 for v in current_alive.values() if v)
            known_active_count = sum(1 for k, v in current_alive.items() if k != "H_novel" and v)

            if known_active_count == 0:
                # All cataloged hypotheses eliminated; only H_novel remains -> escalate
                term_reason = TerminationReason.ALL_CATALOGED_ELIMINATED
                break

            if active_count == 1:
                # Exactly one hypothesis remains -> single survivor found
                term_reason = TerminationReason.SINGLE_SURVIVOR
                break

            if round_idx == max_rounds:
                term_reason = TerminationReason.MAX_ROUNDS_REACHED
                break

            # 8. Append observations to running state for subsequent rounds
            obs_str = json.dumps(observations, ensure_ascii=False)
            state_t = f"{state_t}\n{obs_str}".strip()

        # Compute final verdict probabilities
        if final_verdict_logits is None:
            raise RuntimeError("Model did not execute any forward pass during investigation.")

        probs = self.compute_verdict_probabilities(
            final_verdict_logits,
            hyp_alive=current_alive,
            hyp_keys=hyp_keys,
        )

        winner = max(probs.items(), key=lambda kv: kv[1])[0]
        is_escalated = (term_reason == TerminationReason.ALL_CATALOGED_ELIMINATED) or (winner == "H_novel")

        return InvestigationResult(
            predicted_hypothesis=winner,
            probabilities=probs,
            rounds_executed=len(traces) if traces else 1,
            termination_reason=term_reason,
            is_escalated=is_escalated,
            traces=traces,
        )


def rex_config(model: RexDecisionModel, base_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Generate Rex configuration dictionary compatible with rl_agent_config.json."""
    cfg = dict(base_cfg or {})
    cfg["arch"] = "rex"
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
    scorer, act_head, type_emb, temperature) and initializes a fresh ToolHead.
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

    # Transfer state dict; only tool_head weights should be missing
    rex_state = laya_model.state_dict()
    missing_keys, unexpected_keys = rex_model.load_state_dict(rex_state, strict=False)

    # Validate that only tool_head parameters were missing
    non_tool_missing = [k for k in missing_keys if not k.startswith("tool_head.")]
    if non_tool_missing:
        raise RuntimeError(f"Unexpected missing keys when loading Laya checkpoint: {non_tool_missing}")
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

