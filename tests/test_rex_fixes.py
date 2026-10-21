"""Unit tests verifying foundation bug fixes in PoE and tool masking (Step A0)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from rex.rex_poe import (
    ComparisonOperator,
    FalsificationRule,
    HardMaskingManager,
    PoEFalsificationEngine,
)


def test_poe_supports_all_operators_and_types():
    """Verify PoE engine handles GTE, LTE, IN, NOT_IN, and boolean EQ/NEQ without float cast crashes."""
    engine = PoEFalsificationEngine()

    rules = {
        "H_promo": [
            FalsificationRule("voucher_used", ComparisonOperator.EQ, True),
            FalsificationRule("discount_pct", ComparisonOperator.GTE, 50.0),
            FalsificationRule("user_country", ComparisonOperator.NOT_IN, ["VN", "SG", "TH"]),
        ],
        "H_normal": [
            FalsificationRule("risk_score", ComparisonOperator.LTE, 10.0),
            FalsificationRule("status", ComparisonOperator.IN, ["FLAGGED", "BANNED"]),
        ],
    }

    current_alive = {"H_promo": True, "H_normal": True, "H_novel": True}

    # Case 1: Trigger boolean EQ rule on voucher_used
    obs1 = {"voucher_used": True}
    res1 = engine.evaluate(obs1, current_alive, rules)
    assert not res1["H_promo"]
    assert res1["H_normal"]
    assert res1["H_novel"]  # Residual is always alive

    # Case 2: Categorical NOT_IN rule (e.g. user_country == "US" which is NOT_IN ["VN", "SG", "TH"])
    obs2 = {"user_country": "US"}
    res2 = engine.evaluate(obs2, current_alive, rules)
    assert not res2["H_promo"]

    # Case 3: Categorical IN rule
    obs3 = {"status": "FLAGGED"}
    res3 = engine.evaluate(obs3, current_alive, rules)
    assert not res3["H_normal"]

    # Case 4: GTE numeric rule
    obs4 = {"discount_pct": 50.0}
    res4 = engine.evaluate(obs4, current_alive, rules)
    assert not res4["H_promo"]

    # Case 5: LTE numeric rule
    obs5 = {"risk_score": 9.5}
    res5 = engine.evaluate(obs5, current_alive, rules)
    assert not res5["H_normal"]


def test_tool_mask_prevents_repetition_of_called_tools():
    """Verify HardMaskingManager disables tools already recorded in called_tools."""
    manager = HardMaskingManager()
    tools = ["tool_device", "tool_network", "tool_payment", "tool_general"]
    alive = {"H1": True, "H2": False, "H_novel": True}
    mapping = {
        "tool_device": ["H1"],
        "tool_network": ["H2"],
        "tool_payment": ["H1", "H2"],
        "tool_general": [],  # general tool
    }

    # Turn 1: No tools called yet
    mask1 = manager.compute_tool_mask(tools, alive, mapping, called_tools=[])
    assert mask1 == [True, False, True, True]  # tool_network disabled because H2 is eliminated

    # Turn 2: tool_device and tool_general were called in Turn 1
    mask2 = manager.compute_tool_mask(tools, alive, mapping, called_tools=["tool_device", "tool_general"])
    assert mask2 == [False, False, True, False]  # only tool_payment remains active
