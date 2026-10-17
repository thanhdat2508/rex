"""Multi-task loss functions and training objectives for the Rex Decision Engine.

This module replaces the baseline single-turn cross-entropy objective with an
integrated multi-objective optimization suite:
1. ToolAsymmetricFocalLoss: Multi-label focal loss for parallel tool selection,
   down-weighting dominant negative classes (gamma=2.0) to prevent gradient starvation.
2. PoEMaskedCrossEntropyLoss: Process-of-Elimination (PoE) masked cross-entropy for
   the verdict head, isolating backpropagation strictly to active/surviving hypotheses.
3. ToolOrthogonalLoss: Cosine Gram-matrix regularizer penalizing collinear tool projection
   vectors to prevent avalanche activation (simultaneous 5-10 tool bursts).
4. RexMultiTaskLoss: Composite loss orchestrator returning structured LossBreakdown.
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Named constants to avoid magic numbers (G25)
DEFAULT_FOCAL_GAMMA_NEG: float = 2.0
DEFAULT_FOCAL_ALPHA_POS: float = 1.0
DEFAULT_ORTHO_WEIGHT: float = 0.05
DEFAULT_TOOL_WEIGHT: float = 1.0
DEFAULT_VERDICT_WEIGHT: float = 1.0
MASKED_LOGIT_VALUE: float = -1e4
NUMERICAL_EPSILON: float = 1e-8


@dataclass(frozen=True)
class RexLossWeights:
    """Configurable loss scaling weights for multi-task optimization (F1)."""

    tool_weight: float = DEFAULT_TOOL_WEIGHT
    verdict_weight: float = DEFAULT_VERDICT_WEIGHT
    ortho_weight: float = DEFAULT_ORTHO_WEIGHT


@dataclass(frozen=True)
class LossBreakdown:
    """Structured container holding total and constituent losses for logging and metrics."""

    total_loss: torch.Tensor
    tool_loss: torch.Tensor
    verdict_loss: torch.Tensor
    ortho_loss: torch.Tensor


@dataclass(frozen=True)
class RexMultiTaskInputs:
    """Encapsulates input tensors for multi-task loss computation (F1)."""

    tool_logits: torch.Tensor
    tool_labels: torch.Tensor
    verdict_logits: torch.Tensor
    poe_mask: torch.Tensor
    target_verdict: torch.Tensor
    tool_projection_weight: torch.Tensor


class ToolAsymmetricFocalLoss(nn.Module):
    """Asymmetric Focal Loss for multi-label Tool Selection Head.

    Applies independent sigmoid probability estimation for each tool dimension.
    Applies an exponential decay weight (gamma_neg = 2.0) exclusively on negative
    labels (y = 0) to prevent tens of inactive tools from dominating gradient updates
    over the 1-3 active tools dispatched per round.
    """

    def __init__(
        self,
        gamma_neg: float = DEFAULT_FOCAL_GAMMA_NEG,
        alpha_pos: float = DEFAULT_FOCAL_ALPHA_POS,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma_neg < 0.0:
            raise ValueError(f"gamma_neg must be non-negative, got {gamma_neg}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction '{reduction}'. Expected 'mean', 'sum', or 'none'.")

        self.gamma_neg = gamma_neg
        self.alpha_pos = alpha_pos
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute asymmetric focal loss between tool logits and multi-hot binary targets.

        Args:
            logits: Predicted tool logits of shape [..., num_tools].
            targets: Multi-hot binary labels of shape [..., num_tools] (0.0 or 1.0).

        Returns:
            Reduced scalar loss tensor (or unreduced tensor if reduction='none').
        """
        targets = targets.to(dtype=logits.dtype)

        # Numerically stable sigmoid log-probabilities:
        # log(p) = logsigmoid(z), log(1 - p) = logsigmoid(-z)
        log_p = F.logsigmoid(logits)
        log_not_p = F.logsigmoid(-logits)

        # Probabilities for negative decay factor
        probs = torch.sigmoid(logits)

        # Asymmetric focal weighting:
        # Positive loss retains full gradient (or scaled by alpha_pos)
        pos_loss = -self.alpha_pos * targets * log_p

        # Negative loss is suppressed by (p)^gamma_neg for well-classified negatives
        neg_decay = torch.pow(probs, self.gamma_neg)
        neg_loss = -(1.0 - targets) * neg_decay * log_not_p

        loss = pos_loss + neg_loss

        if self.reduction == "none":
            return loss

        # Sum over tool dimensions, then reduce over batch
        per_sample_loss = loss.sum(dim=-1)
        if self.reduction == "sum":
            return per_sample_loss.sum()
        return per_sample_loss.mean()


class PoEMaskedCrossEntropyLoss(nn.Module):
    """Process-of-Elimination (PoE) Masked Cross-Entropy for Verdict Head.

    Applies hard logit masking (-1e4) to hypotheses that have been eliminated by
    PoE logical criteria (poe_mask == 0). Ensures that softmax normalization and
    gradient backpropagation are computed strictly over surviving candidate hypotheses.
    """

    def __init__(self, masked_logit: float = MASKED_LOGIT_VALUE) -> None:
        super().__init__()
        self.masked_logit = masked_logit

    def forward(
        self,
        logits: torch.Tensor,
        poe_mask: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked cross-entropy loss over surviving hypotheses.

        Args:
            logits: Hypothesis logits of shape [batch_size, num_hypotheses] or [num_hypotheses].
            poe_mask: Binary indicator tensor of shape [batch_size, num_hypotheses],
                      where 1 = active/surviving, 0 = eliminated by PoE.
            targets: Ground-truth target indices of shape [batch_size] or scalar int tensor.

        Returns:
            Scalar cross-entropy loss tensor.
        """
        # Ensure minimum 2D shapes for batch processing
        is_single_sample = logits.dim() == 1
        if is_single_sample:
            logits = logits.unsqueeze(0)
            poe_mask = poe_mask.unsqueeze(0)
            if targets.dim() == 0:
                targets = targets.unsqueeze(0)

        # Hard logit masking for eliminated hypotheses
        masked_logits = logits.masked_fill(poe_mask == 0, self.masked_logit)

        # Cross-entropy automatically computes log-softmax over unmasked survivor logits
        loss = F.cross_entropy(masked_logits, targets)
        return loss


class ToolOrthogonalLoss(nn.Module):
    """Orthogonal Loss on Tool Projection Matrix.

    Penalizes pairwise cosine similarity between row vectors in the tool projection
    weight matrix to enforce directional orthogonality in latent space.
    This prevents avalanche activation (where 5-10 collinear tools trigger simultaneously).
    """

    def __init__(self, epsilon: float = NUMERICAL_EPSILON) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, weight_matrix: torch.Tensor) -> torch.Tensor:
        """Compute off-diagonal Frobenius norm penalty on the Gram matrix.

        Args:
            weight_matrix: Linear layer weight of shape [num_tools, hidden_dim].

        Returns:
            Scalar orthogonal penalty loss tensor.
        """
        num_tools = weight_matrix.size(0)
        if num_tools <= 1:
            return torch.tensor(0.0, device=weight_matrix.device, dtype=weight_matrix.dtype)

        # Normalize each tool vector to unit Euclidean norm
        norm_weights = F.normalize(weight_matrix, p=2, dim=1, eps=self.epsilon)

        # Compute cosine Gram matrix [num_tools, num_tools]
        gram_matrix = torch.matmul(norm_weights, norm_weights.t())

        # Target is the identity matrix (perfect orthogonality across different tools)
        identity = torch.eye(num_tools, device=weight_matrix.device, dtype=weight_matrix.dtype)

        # Squared difference penalizes non-zero off-diagonal correlations
        off_diagonal_diff = gram_matrix - identity
        frobenius_norm_sq = torch.sum(off_diagonal_diff ** 2)

        # Normalize by number of off-diagonal elements: N * (N - 1)
        normalizer = float(num_tools * (num_tools - 1))
        return frobenius_norm_sq / normalizer


class RexMultiTaskLoss(nn.Module):
    """Composite Multi-Task Loss Orchestrator.

    Integrates Tool Focal Loss, PoE-Masked Verdict Cross-Entropy, and Tool Orthogonal Loss
    into a unified multi-objective loss function with structured reporting.
    """

    def __init__(
        self,
        weights: Optional[RexLossWeights] = None,
        focal_gamma_neg: float = DEFAULT_FOCAL_GAMMA_NEG,
        masked_logit: float = MASKED_LOGIT_VALUE,
    ) -> None:
        super().__init__()
        self.weights = weights if weights is not None else RexLossWeights()
        self.tool_loss_fn = ToolAsymmetricFocalLoss(gamma_neg=focal_gamma_neg)
        self.verdict_loss_fn = PoEMaskedCrossEntropyLoss(masked_logit=masked_logit)
        self.ortho_loss_fn = ToolOrthogonalLoss()

    def forward(
        self,
        inputs_or_tool_logits: Union[RexMultiTaskInputs, torch.Tensor],
        tool_labels: Optional[torch.Tensor] = None,
        verdict_logits: Optional[torch.Tensor] = None,
        poe_mask: Optional[torch.Tensor] = None,
        target_verdict: Optional[torch.Tensor] = None,
        tool_projection_weight: Optional[torch.Tensor] = None,
    ) -> LossBreakdown:
        """Compute the weighted composite loss and individual loss breakdown.

        Supports both Dataclass container input (clean code F1) and direct tensor kwargs.
        """
        if isinstance(inputs_or_tool_logits, RexMultiTaskInputs):
            inp = inputs_or_tool_logits
            t_logits = inp.tool_logits
            t_labels = inp.tool_labels
            v_logits = inp.verdict_logits
            p_mask = inp.poe_mask
            v_target = inp.target_verdict
            w_proj = inp.tool_projection_weight
        else:
            t_logits = inputs_or_tool_logits
            t_labels = tool_labels
            v_logits = verdict_logits
            p_mask = poe_mask
            v_target = target_verdict
            w_proj = tool_projection_weight

            if any(x is None for x in (t_labels, v_logits, p_mask, v_target, w_proj)):
                raise ValueError("All constituent tensor arguments must be provided when not using RexMultiTaskInputs.")

        # Compute individual constituent losses
        l_tool = self.tool_loss_fn(t_logits, t_labels)
        l_verdict = self.verdict_loss_fn(v_logits, p_mask, v_target)
        l_ortho = self.ortho_loss_fn(w_proj)

        # Weighted composite loss
        l_total = (
            self.weights.tool_weight * l_tool
            + self.weights.verdict_weight * l_verdict
            + self.weights.ortho_weight * l_ortho
        )

        return LossBreakdown(
            total_loss=l_total,
            tool_loss=l_tool,
            verdict_loss=l_verdict,
            ortho_loss=l_ortho,
        )
