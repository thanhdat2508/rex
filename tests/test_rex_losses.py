"""Unit tests for Rex multi-task loss functions (rex/losses.py).

Tests follow F.I.R.S.T. principles and verify:
1. ToolAsymmetricFocalLoss: gradient suppression on negative labels, numerical stability.
2. PoEMaskedCrossEntropyLoss: strict isolation of backpropagation to surviving hypotheses.
3. ToolOrthogonalLoss: exact zero on orthogonal matrices, positive penalty on collinearity.
4. RexMultiTaskLoss: composite loss weighting, dataclass input compatibility, and end-to-end autograd.
"""

from contextlib import contextmanager
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from rex.losses import (
    DEFAULT_FOCAL_GAMMA_NEG,
    LossBreakdown,
    PoEMaskedCrossEntropyLoss,
    RexLossWeights,
    RexMultiTaskInputs,
    RexMultiTaskLoss,
    ToolAsymmetricFocalLoss,
    ToolOrthogonalLoss,
)


@contextmanager
def assert_raises(exc_type, match=None):
    failed = False
    try:
        yield
    except exc_type as e:
        failed = True
        if match:
            assert match in str(e), f"Expected '{match}' in error message '{str(e)}'"
    assert failed, f"Expected exception {exc_type.__name__} to be raised"


# -----------------------------------------------------------------------------
# Tests for ToolAsymmetricFocalLoss (Step 4.1)
# -----------------------------------------------------------------------------

def test_focal_loss_suppresses_well_classified_negatives():
    """Negative samples predicted with low probability should have loss heavily attenuated by (p)^gamma."""
    loss_fn_focal = ToolAsymmetricFocalLoss(gamma_neg=2.0, reduction="none")
    loss_fn_bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    # High negative confidence (z = -3.0 -> p ~ 0.047)
    logits = torch.tensor([[-3.0]])
    target = torch.tensor([[0.0]])

    focal_val = loss_fn_focal(logits, target).item()
    bce_val = loss_fn_bce(logits, target).item()

    # (p)^2 ~ (0.047)^2 ~ 0.0022 -> focal loss should be ~400x smaller than standard BCE
    assert focal_val < bce_val * 0.01
    assert focal_val >= 0.0


def test_focal_loss_positive_preserves_gradient():
    """Positive targets (y=1) should retain full cross-entropy gradient without negative decay."""
    loss_fn = ToolAsymmetricFocalLoss(reduction="mean")
    logits = torch.tensor([[2.0, -2.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0]])

    loss = loss_fn(logits, targets)
    loss.backward()

    assert logits.grad is not None
    # For y=1 with z=2.0 (p ~ 0.88), grad is p - 1 ~ -0.12 (negative gradient pushes logit up)
    assert logits.grad[0, 0].item() < 0.0
    assert torch.isfinite(logits.grad).all()


def test_focal_loss_numerical_stability_extreme_logits():
    """Extreme logits should not produce NaN or Inf values."""
    loss_fn = ToolAsymmetricFocalLoss()
    extreme_logits = torch.tensor([[-1000.0, 1000.0, 0.0]])
    targets = torch.tensor([[0.0, 1.0, 0.0]])

    loss = loss_fn(extreme_logits, targets)
    assert torch.isfinite(loss)
    assert not torch.isnan(loss)


def test_focal_loss_invalid_arguments_raise():
    """Should raise ValueError on invalid gamma or reduction mode."""
    with assert_raises(ValueError, match="gamma_neg must be non-negative"):
        ToolAsymmetricFocalLoss(gamma_neg=-1.0)

    with assert_raises(ValueError, match="Invalid reduction"):
        ToolAsymmetricFocalLoss(reduction="invalid_mode")


# -----------------------------------------------------------------------------
# Tests for PoEMaskedCrossEntropyLoss (Step 4.2)
# -----------------------------------------------------------------------------

def test_poe_masked_ce_zero_gradient_for_eliminated_hypothesis():
    """Hypotheses with mask=0 should have exactly zero gradient backpropagation."""
    loss_fn = PoEMaskedCrossEntropyLoss()

    # 4 hypotheses: H0 (active), H1 (eliminated by PoE), H2 (active), H3 (novel, active)
    logits = torch.tensor([[1.5, 3.0, 2.0, 0.5]], requires_grad=True)
    poe_mask = torch.tensor([[1, 0, 1, 1]], dtype=torch.long)
    target = torch.tensor([2])  # H2 is ground truth

    loss = loss_fn(logits, poe_mask, target)
    loss.backward()

    assert logits.grad is not None
    # Gradient for eliminated hypothesis H1 MUST be 0.0
    assert logits.grad[0, 1].item() == 0.0
    # Gradient for active target H2 must be negative (encouraging higher logit)
    assert logits.grad[0, 2].item() < 0.0
    assert torch.isfinite(logits.grad).all()


def test_poe_masked_ce_single_and_batch_consistency():
    """Should produce consistent outputs for unbatched (1D) and batched (2D) tensors."""
    loss_fn = PoEMaskedCrossEntropyLoss()

    logits_single = torch.tensor([1.0, 2.0, 0.5])
    mask_single = torch.tensor([1, 1, 0])
    target_single = torch.tensor(1)

    loss_1 = loss_fn(logits_single, mask_single, target_single)

    logits_batch = logits_single.unsqueeze(0)
    mask_batch = mask_single.unsqueeze(0)
    target_batch = target_single.unsqueeze(0)

    loss_2 = loss_fn(logits_batch, mask_batch, target_batch)

    assert torch.allclose(loss_1, loss_2)


# -----------------------------------------------------------------------------
# Tests for ToolOrthogonalLoss (Step 4.3)
# -----------------------------------------------------------------------------

def test_orthogonal_loss_zero_for_identity():
    """A set of mutually orthogonal basis vectors should yield zero orthogonal penalty."""
    loss_fn = ToolOrthogonalLoss()
    # 4 orthonormal tools in 4D space
    ortho_weights = torch.eye(4)

    loss = loss_fn(ortho_weights)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_orthogonal_loss_positive_for_collinear_vectors():
    """Identical or collinear tool projection vectors should produce positive penalty."""
    loss_fn = ToolOrthogonalLoss()
    # 3 identical tool vectors -> maximum collinearity
    collinear_weights = torch.ones(3, 8)

    loss = loss_fn(collinear_weights)
    # Off-diagonal elements of normalized cosine matrix will all be 1.0
    assert loss.item() > 0.0
    assert torch.isclose(loss, torch.tensor(1.0), atol=1e-5)


def test_orthogonal_loss_single_tool():
    """A weight matrix with only 1 tool has no pairs to penalize and should return 0.0."""
    loss_fn = ToolOrthogonalLoss()
    single_weight = torch.randn(1, 16)
    loss = loss_fn(single_weight)
    assert loss.item() == 0.0


# -----------------------------------------------------------------------------
# Tests for RexMultiTaskLoss (Step 4.4)
# -----------------------------------------------------------------------------

def test_rex_multi_task_loss_composition_and_backward():
    """Composite loss should correctly apply weights and support autograd end-to-end."""
    weights = RexLossWeights(tool_weight=2.0, verdict_weight=1.5, ortho_weight=0.1)
    multi_loss = RexMultiTaskLoss(weights=weights)

    batch_size = 2
    num_tools = 4
    num_hypotheses = 3
    hidden_dim = 8

    # Inputs requiring gradients
    tool_logits = torch.randn(batch_size, num_tools, requires_grad=True)
    tool_labels = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]])

    verdict_logits = torch.randn(batch_size, num_hypotheses, requires_grad=True)
    poe_mask = torch.tensor([[1, 0, 1], [1, 1, 1]], dtype=torch.long)
    target_verdict = torch.tensor([0, 1], dtype=torch.long)

    tool_weight_matrix = torch.randn(num_tools, hidden_dim, requires_grad=True)

    inputs = RexMultiTaskInputs(
        tool_logits=tool_logits,
        tool_labels=tool_labels,
        verdict_logits=verdict_logits,
        poe_mask=poe_mask,
        target_verdict=target_verdict,
        tool_projection_weight=tool_weight_matrix,
    )

    breakdown: LossBreakdown = multi_loss(inputs)

    # Check breakdown types and values
    assert isinstance(breakdown, LossBreakdown)
    assert breakdown.total_loss.item() > 0.0
    assert breakdown.tool_loss.item() > 0.0
    assert breakdown.verdict_loss.item() > 0.0
    assert breakdown.ortho_loss.item() >= 0.0

    # Verify weighted sum equation
    expected_total = (
        weights.tool_weight * breakdown.tool_loss
        + weights.verdict_weight * breakdown.verdict_loss
        + weights.ortho_weight * breakdown.ortho_loss
    )
    assert torch.isclose(breakdown.total_loss, expected_total)

    # Run backward pass
    breakdown.total_loss.backward()

    assert tool_logits.grad is not None
    assert verdict_logits.grad is not None
    assert tool_weight_matrix.grad is not None
    assert torch.isfinite(tool_logits.grad).all()
    assert torch.isfinite(verdict_logits.grad).all()
    assert torch.isfinite(tool_weight_matrix.grad).all()


if __name__ == "__main__":
    tests = [
        test_focal_loss_suppresses_well_classified_negatives,
        test_focal_loss_positive_preserves_gradient,
        test_focal_loss_numerical_stability_extreme_logits,
        test_focal_loss_invalid_arguments_raise,
        test_poe_masked_ce_zero_gradient_for_eliminated_hypothesis,
        test_poe_masked_ce_single_and_batch_consistency,
        test_orthogonal_loss_zero_for_identity,
        test_orthogonal_loss_positive_for_collinear_vectors,
        test_orthogonal_loss_single_tool,
        test_rex_multi_task_loss_composition_and_backward,
    ]
    print(f"Running {len(tests)} Rex multi-task loss unit tests...\n")
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("\nAll Rex multi-task loss tests passed successfully!")
