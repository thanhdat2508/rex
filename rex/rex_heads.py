"""Rex Decision Heads: Sufficiency (S), Risk (R), Verdict Bridge, Hypothesis Scorer, and Conformal Calibration."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_SUFFICIENCY_BIAS_INIT: float = -2.0  # sigmoid(-2.0) ~ 0.1192
DEFAULT_HEAD_DROPOUT: float = 0.1


class SufficiencyHead(nn.Module):
    """Head S: Evaluates epistemic sufficiency probability S in [0, 1] from thought state h_T.

    Architecture:
        h_T [B, d] -> Linear(d, hidden_dim) -> LayerNorm -> GELU -> Dropout -> Linear(hidden_dim, 1)

    Notes:
        - Output bias is initialized to -2.0 so an untrained checkpoint does not trigger premature stopping.
        - Returns raw logit [B, 1] for numerical stability under BCEWithLogitsLoss.
    """

    def __init__(
        self,
        d: int = 1024,
        hidden_dim: Optional[int] = None,
        dropout: float = DEFAULT_HEAD_DROPOUT,
    ):
        super().__init__()
        hidden_dim = hidden_dim or (d // 2)
        self.proj_in = nn.Linear(d, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Linear(hidden_dim, 1)

        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.zeros_(self.proj_in.bias)
        nn.init.xavier_uniform_(self.proj_out.weight)
        nn.init.constant_(self.proj_out.bias, DEFAULT_SUFFICIENCY_BIAS_INIT)

    def forward(self, h_T: torch.Tensor) -> torch.Tensor:
        """Compute sufficiency logit of shape [B, 1]."""
        x = self.proj_in(h_T)
        x = self.ln(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.proj_out(x)


class RiskHead(nn.Module):
    """Head R: Estimates operational/regulatory escalation risk R in [0, 1] from thought state h_T.

    Architecture:
        h_T [B, d] -> Linear(d, hidden_dim) -> LayerNorm -> GELU -> Dropout -> Linear(hidden_dim, 1)
    """

    def __init__(
        self,
        d: int = 1024,
        hidden_dim: Optional[int] = None,
        dropout: float = DEFAULT_HEAD_DROPOUT,
    ):
        super().__init__()
        hidden_dim = hidden_dim or (d // 2)
        self.proj_in = nn.Linear(d, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Linear(hidden_dim, 1)

        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.zeros_(self.proj_in.bias)
        nn.init.xavier_uniform_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, h_T: torch.Tensor) -> torch.Tensor:
        """Compute operational risk logit of shape [B, 1]."""
        x = self.proj_in(h_T)
        x = self.ln(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.proj_out(x)


class VerdictBridge(nn.Module):
    """Zero-initialized bridge injecting contextual thought state h_T into hypothesis markers.

    Formula:
        m_tilde_j = m_j + g_v * (W_v * h_T)
        logit_j = Scorer(m_tilde_j)

    Properties:
        - At initialization, W_v = 0 and g_v = 0 -> delta = 0 identically.
        - Preserves 100% regression parity with base Laya/Rex1 checkpoints on turn 1.
    """

    def __init__(self, d: int = 1024):
        super().__init__()
        self.d = d
        self.w_v = nn.Linear(d, d, bias=False)
        self.g_v = nn.Parameter(torch.zeros(1))

        # Explicit zero initialization guarantees zero perturbation at checkpoint load
        nn.init.zeros_(self.w_v.weight)

    def forward(self, hyp_markers: torch.Tensor, h_T: torch.Tensor) -> torch.Tensor:
        """Inject thought state h_T into hypothesis markers.

        Args:
            hyp_markers: Marker representations [B, K, d] extracted from [MASK] tokens.
            h_T: Contextualized thought state [B, d].

        Returns:
            Enhanced markers [B, K, d] ready for DecisionModel.scorer.
        """
        h_proj = self.w_v(h_T).unsqueeze(1)    # [B, 1, d]
        delta = self.g_v * h_proj              # [B, 1, d] broadcasts across K
        return hyp_markers + delta


class HypothesisScorer(nn.Module):
    """Neural hypothesis scorer estimating survival likelihood on evidence memory M for PoE.

    Formula:
        Score(H_j) = sigmoid(FFN(CrossAttn(e_{H_j}, LayerNorm(M)))) in [0, 1]
    """

    def __init__(self, d: int = 1024, nhead: int = 8, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or (d // 2)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d, num_heads=nhead, batch_first=True)
        self.ln_m = nn.LayerNorm(d)
        self.ln_q = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        nn.init.xavier_uniform_(self.ffn[0].weight)
        nn.init.zeros_(self.ffn[0].bias)
        nn.init.xavier_uniform_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(
        self,
        hyp_embeddings: torch.Tensor,
        M: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute hypothesis survival scores in [0, 1].

        Args:
            hyp_embeddings: Hypothesis embedding vectors [B, K, d] or [K, d].
            M: Evidence tokens [B, L_mem, d].
            memory_mask: Boolean mask [B, L_mem] where True is valid, False is padding.

        Returns:
            Survival scores [B, K] in range [0, 1].
        """
        if hyp_embeddings.dim() == 2:
            hyp_embeddings = hyp_embeddings.unsqueeze(0).expand(M.size(0), -1, -1)

        if M.size(1) == 0:
            # Empty evidence -> prior score 0.5 across all hypotheses
            return torch.full((hyp_embeddings.size(0), hyp_embeddings.size(1)), 0.5, device=M.device)

        q = self.ln_q(hyp_embeddings)
        kv = self.ln_m(M)
        pad_mask = ~memory_mask.bool() if memory_mask is not None else None

        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv, key_padding_mask=pad_mask)
        scores = torch.sigmoid(self.ffn(attn_out).squeeze(-1))
        return scores


def compute_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    """Vectorized Expected Calibration Error (ECE) calculation.

    Args:
        probs: Predicted positive probabilities of shape [N] in [0, 1].
        labels: Binary ground-truth labels of shape [N] (0 or 1).
        n_bins: Number of confidence bins (default: 10).

    Returns:
        Scalar ECE metric as float.
    """
    probs = probs.detach().view(-1)
    labels = labels.detach().view(-1).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = torch.tensor(0.0, device=probs.device)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (probs >= bin_lower) & (probs < bin_upper) if i < n_bins - 1 else (probs >= bin_lower) & (probs <= bin_upper)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            accuracy_in_bin = labels[in_bin].mean()
            avg_confidence_in_bin = probs[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece.item())


class TemperatureScaling(nn.Module):
    """Learnable temperature scaling for post-hoc probability calibration."""

    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by clamped temperature."""
        temp = torch.clamp(self.temperature, min=0.1, max=10.0)
        return logits / temp


class ConformalCalibrator:
    """Split-conformal prediction set generator providing distribution-free coverage via Adaptive Prediction Sets (APS)."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.q_hat: Optional[float] = None

    def calibrate(self, probs: torch.Tensor, targets: torch.Tensor) -> float:
        """Calibrate cumulative non-conformity threshold q_hat on held-out calibration split.

        Args:
            probs: Predicted class probabilities of shape [N, K].
            targets: True class index tensor of shape [N].

        Returns:
            Calibrated quantile threshold q_hat.
        """
        n = probs.size(0)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        # Find where true target appears in sorted order
        target_ranks = (sorted_indices == targets.unsqueeze(1)).nonzero(as_tuple=True)[1]
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        scores = cum_probs.gather(1, target_ranks.unsqueeze(1)).squeeze(1)

        # Finite-sample adjusted empirical quantile
        quantile_val = math.ceil((n + 1) * (1.0 - self.alpha)) / n
        quantile_val = min(1.0, max(0.0, quantile_val))

        self.q_hat = float(torch.quantile(scores, quantile_val).item())
        return self.q_hat

    def predict_set(
        self,
        probs: torch.Tensor,
        labels: Optional[List[str]] = None,
        q_hat: Optional[float] = None,
    ) -> List[List[str]]:
        """Construct conformal prediction sets for batch samples using cumulative probability coverage.

        Args:
            probs: Predicted probabilities of shape [B, K].
            labels: List of K class names.
            q_hat: Threshold value (falls back to calibrated self.q_hat or 1 - alpha).

        Returns:
            List of prediction set string labels per sample.
        """
        threshold = q_hat if q_hat is not None else (self.q_hat if self.q_hat is not None else (1.0 - self.alpha))
        b, k = probs.shape
        names = labels if labels is not None and len(labels) == k else [f"H_{i}" for i in range(k)]

        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        sets: List[List[str]] = []

        for i in range(b):
            row_probs = sorted_probs[i]
            row_indices = sorted_indices[i]
            selected: List[str] = []
            cum_p = 0.0

            for j in range(k):
                selected.append(names[row_indices[j].item()])
                cum_p += row_probs[j].item()
                if cum_p >= threshold:
                    break

            sets.append(selected)

        return sets
