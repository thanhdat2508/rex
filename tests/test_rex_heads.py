"""Unit tests for Rex decision heads, calibration, and conformal prediction (Step A3)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from rex.rex_heads import (
    ConformalCalibrator,
    HypothesisScorer,
    RiskHead,
    SufficiencyHead,
    TemperatureScaling,
    VerdictBridge,
    compute_ece,
)


def test_sufficiency_head_bias_init_and_shapes():
    """Verify Head S outputs [B, 1] with initial prior probability ~0.1192."""
    b, d = 10, 256
    head_s = SufficiencyHead(d=d, hidden_dim=64)
    head_s.eval()

    # Exact prior check at zero state
    zero_logit = head_s(torch.zeros(1, d))
    zero_prob = torch.sigmoid(zero_logit).item()
    assert abs(zero_prob - 0.1192) < 0.01

    # Random input check
    h_T = torch.randn(b, d)
    logit = head_s(h_T)
    assert logit.shape == (b, 1)

    prob = torch.sigmoid(logit)
    # Average across batch should remain low around the prior
    assert prob.mean().item() < 0.35


def test_risk_head_shapes_and_backward():
    """Verify Head R outputs [B, 1] and gradients flow back through h_T."""
    b, d = 2, 128
    head_r = RiskHead(d=d)
    h_T = torch.randn(b, d, requires_grad=True)

    logit = head_r(h_T)
    assert logit.shape == (b, 1)

    loss = logit.sum()
    loss.backward()
    assert h_T.grad is not None


def test_verdict_bridge_zero_init_parity():
    """Verify VerdictBridge with W_v=0, g_v=0 produces exact numerical identity."""
    b, k, d = 2, 4, 128
    bridge = VerdictBridge(d=d)
    markers = torch.randn(b, k, d)
    h_T = torch.randn(b, d)

    enhanced = bridge(markers, h_T)
    # Must be exactly identical with zero delta
    torch.testing.assert_close(enhanced, markers, rtol=0, atol=0)


def test_hypothesis_scorer_shapes_and_bounds():
    """Verify HypothesisScorer produces survival scores in [0, 1]."""
    b, k, d, l_mem = 2, 3, 128, 10
    scorer = HypothesisScorer(d=d, nhead=2)
    hyp_emb = torch.randn(b, k, d)
    M = torch.randn(b, l_mem, d)
    mask = torch.ones(b, l_mem, dtype=torch.bool)

    scores = scorer(hyp_emb, M, memory_mask=mask)
    assert scores.shape == (b, k)
    assert (scores >= 0.0).all() and (scores <= 1.0).all()


def test_compute_ece_calculation():
    """Verify ECE calculation on calibrated vs miscalibrated predictions."""
    # Perfectly calibrated case: probs 0.8 on all-ones labels has 0.2 calibration error
    probs = torch.full((100,), 0.8)
    labels = torch.ones(100)
    ece = compute_ece(probs, labels, n_bins=10)
    assert abs(ece - 0.2) < 0.05


def test_conformal_calibrator_prediction_sets():
    """Verify ConformalCalibrator computes q_hat and generates valid non-empty sets."""
    calibrator = ConformalCalibrator(alpha=0.1)

    # 100 calibration samples over 3 classes
    cal_probs = torch.tensor([[0.8, 0.1, 0.1]] * 100)
    cal_targets = torch.zeros(100, dtype=torch.long)

    q_hat = calibrator.calibrate(cal_probs, cal_targets)
    assert abs(q_hat - 0.8) < 0.05  # cumulative mass threshold ~0.8

    test_probs = torch.tensor([
        [0.85, 0.10, 0.05],  # unambiguous -> single set ["H_0"]
        [0.40, 0.40, 0.20],  # ambiguous -> multi-label set
    ])
    sets = calibrator.predict_set(test_probs, labels=["H_0", "H_1", "H_2"])
    assert len(sets) == 2
    assert sets[0] == ["H_0"]
    assert len(sets[1]) >= 2
