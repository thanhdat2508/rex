"""Unit tests for HybridPoEEngine, neural score gating, and revival (Step A6)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from rex.rex_poe import (
    ComparisonOperator,
    FalsificationRule,
    HybridPoEEngine,
    apply_poe_logit_mask,
)


def test_hybrid_poe_symbolic_and_neural_combination():
    """Verify combination of symbolic rule veto and neural score elimination."""
    engine = HybridPoEEngine(default_theta_elim=0.20)
    hyp_keys = ["H_account_takeover", "H_friendly_fraud", "H_stolen_card", "H_novel"]

    # Rule: friendly fraud is falsified if card_present == False
    rules = {
        "H_friendly_fraud": [
            FalsificationRule("card_present", ComparisonOperator.EQ, False)
        ]
    }
    obs = {"card_present": False}

    # Case 1:
    # - H_account_takeover: neural score 0.85 (survives)
    # - H_friendly_fraud: neural score 0.90 (high, BUT falsified by card_present == False rule!)
    # - H_stolen_card: neural score 0.05 (< 0.20, eliminated by neural threshold)
    # - H_novel: residual slot (must survive)
    neural_scores = torch.tensor([[0.85, 0.90, 0.05, 0.50]])
    alive = engine.evaluate(neural_scores, hyp_keys, obs, rules)

    assert alive["H_account_takeover"] is True
    assert alive["H_friendly_fraud"] is False  # Rule veto
    assert alive["H_stolen_card"] is False     # Neural score below theta_elim
    assert alive["H_novel"] is True            # Always True invariant


def test_hypothesis_revival_across_rounds():
    """Verify hypothesis eliminated by temporary low neural score can revive in later rounds."""
    engine = HybridPoEEngine(default_theta_elim=0.20)
    hyp_keys = ["H_device_fraud", "H_novel"]
    rules = {}
    obs = {}

    # Round 1: neural score 0.10 (< 0.20) -> eliminated
    alive_r1 = engine.evaluate(torch.tensor([[0.10, 0.5]]), hyp_keys, obs, rules)
    assert alive_r1["H_device_fraud"] is False

    # Round 2: new evidence clarifies doubt, neural score jumps to 0.75 -> revived!
    alive_r2 = engine.evaluate(torch.tensor([[0.75, 0.5]]), hyp_keys, obs, rules)
    assert alive_r2["H_device_fraud"] is True


def test_apply_poe_logit_mask():
    """Verify logit masking preserves active logits and penalizes eliminated by -1e4."""
    logits = torch.tensor([[2.5, 1.0, 3.0]])
    hyp_alive = torch.tensor([[True, False, True]])

    masked = apply_poe_logit_mask(logits, hyp_alive, masked_val=-1e4)
    assert masked[0, 0].item() == 2.5
    assert masked[0, 1].item() <= -9990.0  # ~ -10000 + 1.0 = -9999
    assert masked[0, 2].item() == 3.0
