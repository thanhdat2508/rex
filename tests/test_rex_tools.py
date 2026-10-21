"""Unit tests for ToolMatcherHead, ToolGainHead, and 3-branch epistemic selection (Step A5)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from rex.rex_tools import ToolGainHead, ToolMatcherHead, dispatch_three_branch


def test_tool_matcher_head_shapes_and_projection_weight():
    """Verify ToolMatcherHead outputs logits and provides [N_tools, 256] orthogonal weights."""
    b, d, n_tools = 2, 128, 5
    matcher = ToolMatcherHead(d=d, proj_dim=64)

    # Initialize tool description embeddings
    tool_desc_emb = torch.randn(n_tools, d)
    matcher.set_tool_embeddings(tool_desc_emb)

    h_T = torch.randn(b, d)
    logits = matcher(h_T)
    assert logits.shape == (b, n_tools)

    proj_weight = matcher.tool_projection_weight
    assert proj_weight.shape == (n_tools, 64)
    # Check unit L2 norm
    norms = torch.norm(proj_weight, p=2, dim=-1)
    torch.testing.assert_close(norms, torch.ones(n_tools), rtol=1e-5, atol=1e-5)


def test_tool_gain_head_shapes_and_bounds():
    """Verify ToolGainHead produces information gain estimates strictly bounded in [0, 1]."""
    b, d, n_tools = 3, 128, 4
    gain_head = ToolGainHead(d=d, hidden_dim=64)

    h_T = torch.randn(b, d)
    tool_emb = torch.randn(n_tools, d)

    gains = gain_head(h_T, tool_emb)
    assert gains.shape == (b, n_tools)
    assert (gains >= 0.0).all() and (gains <= 1.0).all()


def test_dispatch_three_branch_logic():
    """Verify 3-branch decision logic: CALL_TOOL, VERDICT, and STALL."""
    tool_names = ["tool_read", "tool_light", "tool_heavy"]
    tool_levels = {"tool_read": "READ", "tool_light": "LIGHT", "tool_heavy": "HEAVY"}
    tool_costs = {"tool_read": 0.0, "tool_light": 0.2, "tool_heavy": 0.5}

    tool_logits = torch.tensor([[1.0, 0.5, 0.2]])

    # Case 1: Branch 1 - High gain on light tool -> CALL_TOOL
    gains_high = torch.tensor([[0.05, 0.8, 0.1]])
    action1, selected1, _ = dispatch_three_branch(
        tool_names=tool_names,
        tool_logits=tool_logits,
        tool_gains=gains_high,
        s_prob=torch.tensor(0.4),
        r_prob=torch.tensor(0.1),
        round_idx=2,
        tool_levels=tool_levels,
        tool_costs=tool_costs,
    )
    assert action1 == "CALL_TOOL"
    assert "tool_light" in selected1

    # Case 2: Branch 2 - All gains low, S is high (0.90), R is low (0.10) -> VERDICT
    gains_low = torch.tensor([[0.0, 0.05, 0.05]])
    action2, selected2, _ = dispatch_three_branch(
        tool_names=tool_names,
        tool_logits=tool_logits,
        tool_gains=gains_low,
        s_prob=torch.tensor(0.92),
        r_prob=torch.tensor(0.15),
        round_idx=2,
        called_tools=["tool_read"],
        tool_levels=tool_levels,
        tool_costs=tool_costs,
    )
    assert action2 == "VERDICT"
    assert selected2 == []

    # Case 3: Branch 3 - All gains low, S is low (0.50) -> STALL
    action3, selected3, _ = dispatch_three_branch(
        tool_names=tool_names,
        tool_logits=tool_logits,
        tool_gains=gains_low,
        s_prob=torch.tensor(0.50),
        r_prob=torch.tensor(0.15),
        round_idx=2,
        called_tools=["tool_read"],
        tool_levels=tool_levels,
        tool_costs=tool_costs,
    )
    assert action3 == "STALL"
    assert selected3 == []


def test_round_1_exploratory_guardrail():
    """Verify round 1 always attempts at least 1 cheap READ tool even if model predicts STALL."""
    tool_names = ["tool_read", "tool_heavy"]
    tool_levels = {"tool_read": "READ", "tool_heavy": "HEAVY"}
    tool_costs = {"tool_read": 0.0, "tool_heavy": 0.5}

    tool_logits = torch.tensor([[1.5, 0.2]])
    gains_zero = torch.tensor([[0.0, 0.0]])  # all gains low

    action, selected, _ = dispatch_three_branch(
        tool_names=tool_names,
        tool_logits=tool_logits,
        tool_gains=gains_zero,
        s_prob=torch.tensor(0.30),
        r_prob=torch.tensor(0.10),
        round_idx=1,  # Turn 1
        tool_levels=tool_levels,
        tool_costs=tool_costs,
    )
    assert action == "CALL_TOOL"
    assert selected == ["tool_read"]
