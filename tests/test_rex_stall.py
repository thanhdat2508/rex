"""Unit tests for learned STALL signal and two safety buffers (Step A7)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from rex.rex_tools import evaluate_stall_and_dispatch


def test_stall_buffer_1_last_resort_cheap_read():
    """Verify STALL candidate attempts uncalled cheap read tool before escalating."""
    tool_names = ["tool_read", "tool_heavy"]
    tool_levels = {"tool_read": "READ", "tool_heavy": "HEAVY"}
    tool_costs = {"tool_read": 0.1, "tool_heavy": 0.5}

    tool_logits = torch.tensor([[1.0, 0.2]])
    gains_low = torch.tensor([[0.02, 0.01]])  # both below their costs (0.02 < 0.1, 0.01 < 0.5)

    # Candidate for STALL: S=0.50, R=0.10, but tool_read has never been called and last_read_tried=False
    action, selected, stall, mode, _, last_read_out, _ = evaluate_stall_and_dispatch(
        tool_names=tool_names,
        tool_logits=tool_logits,
        tool_gains=gains_low,
        s_prob=torch.tensor(0.50),
        r_prob=torch.tensor(0.10),
        round_idx=2,
        called_tools=[],
        last_read_tried=False,
        tool_levels=tool_levels,
        tool_costs=tool_costs,
    )
    # Buffer 1 intercept
    assert action == "CALL_TOOL"
    assert selected == ["tool_read"]
    assert not stall
    assert last_read_out is True


def test_stall_buffer_2_conformal_singleton_auto_commit():
    """Verify STALL candidate with singleton conformal set and low risk auto-commits to VERDICT."""
    tool_names = ["tool_read", "tool_heavy"]
    tool_levels = {"tool_read": "READ", "tool_heavy": "HEAVY"}
    tool_costs = {"tool_read": 0.0, "tool_heavy": 0.5}

    tool_logits = torch.tensor([[1.0, 0.2]])
    gains_low = torch.tensor([[0.0, 0.0]])

    # Candidate for STALL: S=0.60, R=0.10 <= 0.30, tool_read already called, conformal_set is singleton ["H_stolen"]
    action, selected, stall, mode, _, _, _ = evaluate_stall_and_dispatch(
        tool_names=tool_names,
        tool_logits=tool_logits,
        tool_gains=gains_low,
        s_prob=torch.tensor(0.60),
        r_prob=torch.tensor(0.10),
        round_idx=2,
        called_tools=["tool_read"],
        conformal_set=["H_stolen"],  # Singleton set
        last_read_tried=True,
        tool_levels=tool_levels,
        tool_costs=tool_costs,
    )
    # Buffer 2 intercept
    assert action == "VERDICT"
    assert selected == []
    assert not stall


def test_stall_emitted_when_buffers_exhausted():
    """Verify STALL is emitted when both buffers are exhausted and certainty is insufficient."""
    tool_names = ["tool_read", "tool_heavy"]
    tool_levels = {"tool_read": "READ", "tool_heavy": "HEAVY"}
    tool_costs = {"tool_read": 0.0, "tool_heavy": 0.5}

    tool_logits = torch.tensor([[1.0, 0.2]])
    gains_low = torch.tensor([[0.0, 0.0]])

    # Both buffers exhausted: tool_read already called, conformal set is ambiguous multi-label ["H1", "H2"]
    action, selected, stall, mode, rule_stall, _, _ = evaluate_stall_and_dispatch(
        tool_names=tool_names,
        tool_logits=tool_logits,
        tool_gains=gains_low,
        s_prob=torch.tensor(0.55),
        r_prob=torch.tensor(0.15),
        round_idx=2,
        called_tools=["tool_read"],
        conformal_set=["H_1", "H_2"],
        last_read_tried=True,
        s_prev=0.53,  # delta = 0.02 < 0.05 -> rule_stall is True
        epsilon=0.05,
        tool_levels=tool_levels,
        tool_costs=tool_costs,
    )
    assert action == "STALL"
    assert selected == []
    assert stall is True
    assert mode == "LEARNED"
    assert rule_stall is True
