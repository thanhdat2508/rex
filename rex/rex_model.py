"""Rex model architecture: RexDecisionModel, ToolHead, and checkpoint utilities."""
import math
from contextlib import nullcontext
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn

from .common import DecisionModel


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

