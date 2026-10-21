"""Tool Matcher, Epistemic Information Gain Head, and Three-Branch Dispatcher for Rex (Step A5)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_PROJ_DIM: int = 256
DEFAULT_MATCHER_TEMPERATURE: float = 0.07
DEFAULT_BUDGET_CAPS: Tuple[int, ...] = (3, 2, 1)
DEFAULT_S_THRESHOLD: float = 0.85
DEFAULT_R_THRESHOLD: float = 0.30


class ToolMatcherHead(nn.Module):
    """Zero-shot Tool Matcher mapping contextual thought state h_T to tool relevance logits.

    Architecture:
        u = normalize(W_q * h_T) in R^256
        V = normalize(W_t * e_tool) in R^{N_tools x 256}
        tool_logits = (u @ V.T) / tau
    """

    def __init__(
        self,
        d: int = 1024,
        proj_dim: int = DEFAULT_PROJ_DIM,
        temperature: float = DEFAULT_MATCHER_TEMPERATURE,
    ):
        super().__init__()
        self.d = d
        self.proj_dim = proj_dim
        self.temperature = temperature

        self.proj_query = nn.Linear(d, proj_dim, bias=False)
        self.proj_tool = nn.Linear(d, proj_dim, bias=False)

        nn.init.xavier_uniform_(self.proj_query.weight)
        nn.init.xavier_uniform_(self.proj_tool.weight)

        self.register_buffer("tool_desc_embeddings", None)

    def set_tool_embeddings(self, embeddings: torch.Tensor) -> None:
        """Cache normalized tool description embeddings of shape [N_tools, d]."""
        self.tool_desc_embeddings = F.normalize(embeddings, p=2, dim=-1)

    @property
    def tool_projection_weight(self) -> torch.Tensor:
        """Projection matrix [N_tools, proj_dim] for ToolOrthogonalLoss compatibility."""
        if self.tool_desc_embeddings is None:
            raise RuntimeError("Tool embeddings must be initialized via set_tool_embeddings() before accessing weights.")
        v = self.proj_tool(self.tool_desc_embeddings)
        return F.normalize(v, p=2, dim=-1)

    @property
    def tool_embeddings(self) -> torch.Tensor:
        """Alias for tool_projection_weight for backward compatibility with loss inputs."""
        return self.tool_projection_weight

    @property
    def proj_out(self) -> nn.Linear:
        """Alias for proj_tool for backward compatibility with optimizer gradient checks."""
        return self.proj_tool

    def forward(self, h_T: torch.Tensor) -> torch.Tensor:
        """Compute tool relevance logits [B, N_tools]."""
        if self.tool_desc_embeddings is None:
            raise RuntimeError("set_tool_embeddings() must be called prior to ToolMatcherHead forward pass.")

        u = F.normalize(self.proj_query(h_T), p=2, dim=-1)  # [B, proj_dim]
        V = self.tool_projection_weight                      # [N_tools, proj_dim]
        logits = torch.matmul(u, V.t()) / self.temperature   # [B, N_tools]
        return logits.float()


class ToolGainHead(nn.Module):
    """Head estimating expected epistemic information gain in [0, 1] per tool.

    Architecture:
        gain_k = sigmoid(MLP([h_T ; e_tool_k ; h_T * e_tool_k]))
    """

    def __init__(self, d: int = 1024, hidden_dim: int = 256):
        super().__init__()
        self.d = d
        self.mlp = nn.Sequential(
            nn.Linear(3 * d, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.xavier_uniform_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, h_T: torch.Tensor, tool_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute information gain estimates in [0, 1].

        Args:
            h_T: Thought state tensor [B, d].
            tool_embeddings: Precomputed tool representations [N_tools, d].

        Returns:
            Expected gain tensor of shape [B, N_tools] in range [0, 1].
        """
        B, d = h_T.shape
        N = tool_embeddings.size(0)

        h_exp = h_T.unsqueeze(1).expand(-1, N, -1)
        t_exp = tool_embeddings.unsqueeze(0).expand(B, -1, -1)
        feat = torch.cat([h_exp, t_exp, h_exp * t_exp], dim=-1)  # [B, N, 3*d]
        return torch.sigmoid(self.mlp(feat).squeeze(-1))         # [B, N]


def dispatch_three_branch(
    tool_names: List[str],
    tool_logits: torch.Tensor,
    tool_gains: torch.Tensor,
    s_prob: torch.Tensor,
    r_prob: torch.Tensor,
    round_idx: int,
    called_tools: Optional[List[str]] = None,
    tool_levels: Optional[Dict[str, str]] = None,
    tool_costs: Optional[Dict[str, float]] = None,
    tool_mask: Optional[torch.Tensor] = None,
    budget_caps: Tuple[int, ...] = DEFAULT_BUDGET_CAPS,
    s_threshold: float = DEFAULT_S_THRESHOLD,
    r_threshold: float = DEFAULT_R_THRESHOLD,
    gain_head_trained: bool = True,
) -> Tuple[str, List[str], Dict[str, float]]:
    """Execute 3-branch epistemic tool selection and stopping logic.

    Branch 1: If any qualifying tool has gain >= tau_k -> CALL_TOOL (Top-K_t).
    Branch 2: If no tool qualifies, and S >= 0.85 and R <= tau_R -> VERDICT.
    Branch 3: If no tool qualifies, and S < 0.85 -> STALL.

    Returns:
        Tuple of (proposed_action, selected_tools, effective_gains).
    """
    called_set = set(called_tools or [])
    levels = tool_levels or {}
    costs = tool_costs or {}
    n_tools = len(tool_names)

    cap_idx = max(0, min(round_idx - 1, len(budget_caps) - 1))
    budget_k = budget_caps[cap_idx]

    # Convert tensors to 1D flat arrays
    gains_flat = tool_gains.squeeze(0).tolist() if tool_gains.dim() > 1 else tool_gains.tolist()
    logits_flat = tool_logits.squeeze(0).tolist() if tool_logits.dim() > 1 else tool_logits.tolist()
    s_val = float(s_prob.item())
    r_val = float(r_prob.item())

    # Build effective gains and qualifying candidate list
    effective_gains: Dict[str, float] = {}
    qualifying_tools: List[Tuple[str, float, float]] = []  # (name, gain, score)

    for i, name in enumerate(tool_names):
        # Environmental masking: previously called tools get gain = 0.0
        if name in called_set:
            effective_gains[name] = 0.0
            continue

        # Hard PoE masking: if tool_mask is False, disable
        if tool_mask is not None:
            mask_val = tool_mask[i].item() if tool_mask.dim() == 1 else tool_mask[0, i].item()
            if not mask_val:
                effective_gains[name] = 0.0
                continue

        g_i = gains_flat[i]
        effective_gains[name] = g_i

        # Threshold derivation: cheap READ tools have tau_k = 0.0
        level = levels.get(name, "READ")
        cost = costs.get(name, 0.0)
        tau_k = 0.0 if (level == "READ" and cost <= 0.0) else min(0.5, cost)

        if not gain_head_trained or g_i >= tau_k:
            qualifying_tools.append((name, g_i, logits_flat[i]))

    # If gain head not trained, bypass gain filtering and select top-K by logits
    if not gain_head_trained:
        if qualifying_tools:
            qualifying_tools.sort(key=lambda x: -x[2])  # sort by logits
            selected = [t[0] for t in qualifying_tools[:budget_k]]
            return "CALL_TOOL", selected, effective_gains
        return "VERDICT" if s_val >= s_threshold else "STALL", [], effective_gains

    # Branch 1: At least one tool has gain >= tau_k
    if qualifying_tools:
        # Sort descending by gain, tie-break by relevance logit
        qualifying_tools.sort(key=lambda x: (-x[1], -x[2]))
        selected = [t[0] for t in qualifying_tools[:budget_k]]
        return "CALL_TOOL", selected, effective_gains

    # Round 1 Exploratory Guardrail: Round 1 must attempt at least 1 uncalled cheap READ tool
    if round_idx == 1:
        uncalled_read = [
            name for name in tool_names
            if name not in called_set and levels.get(name, "READ") == "READ"
        ]
        if uncalled_read:
            # Pick highest relevance logit among uncalled read tools
            uncalled_read.sort(key=lambda t: -logits_flat[tool_names.index(t)])
            return "CALL_TOOL", [uncalled_read[0]], effective_gains

    # Branch 2: S >= s_threshold and R <= r_threshold -> VERDICT
    if s_val >= s_threshold and r_val <= r_threshold:
        return "VERDICT", [], effective_gains

    # Branch 3: S < s_threshold -> STALL
    return "STALL", [], effective_gains


def evaluate_stall_and_dispatch(
    tool_names: List[str],
    tool_logits: torch.Tensor,
    tool_gains: torch.Tensor,
    s_prob: torch.Tensor,
    r_prob: torch.Tensor,
    round_idx: int,
    called_tools: Optional[List[str]] = None,
    conformal_set: Optional[List[str]] = None,
    last_read_tried: bool = False,
    tool_levels: Optional[Dict[str, str]] = None,
    tool_costs: Optional[Dict[str, float]] = None,
    tool_mask: Optional[torch.Tensor] = None,
    budget_caps: Tuple[int, ...] = DEFAULT_BUDGET_CAPS,
    s_threshold: float = DEFAULT_S_THRESHOLD,
    r_threshold: float = DEFAULT_R_THRESHOLD,
    gain_head_trained: bool = True,
    s_prev: Optional[float] = None,
    epsilon: float = 0.05,
) -> Tuple[str, List[str], bool, str, bool, bool, Dict[str, float]]:
    """Comprehensive tool selection and STALL evaluation with two safety buffers (Step A7).

    Safety Buffers before emitting final STALL:
        Buffer 1 (Last-resort cheap read): If STALL condition is met, but an uncalled cheap
                 READ tool exists and has not yet been tried as a last resort, dispatch it
                 and mark last_read_tried = True.
        Buffer 2 (Conformal singleton auto-commit): If STALL condition is met, but conformal
                 prediction set is a singleton (|C| == 1) and risk R <= tau_R, auto-commit
                 to VERDICT instead of escalating to human.

    Returns:
        Tuple of:
            - proposed_action: "CALL_TOOL", "VERDICT", or "STALL"
            - selected_tools: List[str]
            - stall: bool
            - stall_mode: "LEARNED" or "NONE"
            - rule_stall: bool
            - last_read_tried_out: bool
            - effective_gains: Dict[str, float]
    """
    s_val = float(s_prob.item())
    r_val = float(r_prob.item())
    levels = tool_levels or {}
    called_set = set(called_tools or [])

    # Evaluate base 3-branch selection
    action, selected, effective_gains = dispatch_three_branch(
        tool_names=tool_names,
        tool_logits=tool_logits,
        tool_gains=tool_gains,
        s_prob=s_prob,
        r_prob=r_prob,
        round_idx=round_idx,
        called_tools=called_tools,
        tool_levels=tool_levels,
        tool_costs=tool_costs,
        tool_mask=tool_mask,
        budget_caps=budget_caps,
        s_threshold=s_threshold,
        r_threshold=r_threshold,
        gain_head_trained=gain_head_trained,
    )

    # Compute rule_stall metric (S delta < epsilon over rounds) for telemetry comparison
    rule_stall = False
    if s_prev is not None and round_idx >= 2:
        rule_stall = (s_val - s_prev) < epsilon

    # If action is not STALL, return directly
    if action != "STALL":
        return action, selected, False, "NONE", rule_stall, last_read_tried, effective_gains

    # Action is candidate for STALL -> Apply 2 Safety Buffers
    # Safety Buffer 1: Last-resort cheap READ attempt
    if not last_read_tried:
        uncalled_read = [
            t for t in tool_names
            if t not in called_set and levels.get(t, "READ") == "READ"
        ]
        if uncalled_read:
            # Dispatch the uncalled read tool with highest relevance logit
            logits_flat = tool_logits.squeeze(0).tolist() if tool_logits.dim() > 1 else tool_logits.tolist()
            uncalled_read.sort(key=lambda t: -logits_flat[tool_names.index(t)])
            return "CALL_TOOL", [uncalled_read[0]], False, "NONE", rule_stall, True, effective_gains

    # Safety Buffer 2: High-certainty conformal auto-commit (|C| == 1 and R <= tau_R)
    if conformal_set is not None and len(conformal_set) == 1 and r_val <= r_threshold:
        return "VERDICT", [], False, "NONE", rule_stall, last_read_tried, effective_gains

    # If both buffers fail -> emit learned STALL signal
    return "STALL", [], True, "LEARNED", rule_stall, last_read_tried, effective_gains
