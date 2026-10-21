"""Unit tests for RexState, RexStepOutput, and trace serialization (Step A1)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from rex.rex_types import RexState, RexStepOutput


def test_rex_state_initialization_and_empty():
    """Verify RexState.empty creates valid tensors with zero elements in memory."""
    state = RexState.empty(batch_size=2, d=1024)
    assert state.h.shape == (2, 1024)
    assert state.M.shape == (2, 0, 1024)
    assert state.memory_mask.shape == (2, 0)
    assert state.round_idx == 1
    assert len(state.called_tools) == 2
    assert not state.last_read_tried
    assert not state.is_truncated


def test_rex_step_output_to_trace_dict_serialization():
    """Verify to_trace_dict produces strictly primitive Python types without any torch.Tensor."""
    b = 2
    n_tools = 4
    k_hyps = 3

    out = RexStepOutput(
        s_logit=torch.tensor([[1.2], [-0.5]]),
        s_prob=torch.tensor([[0.7685], [0.3775]]),
        r_logit=torch.tensor([[-0.1], [0.8]]),
        r_prob=torch.tensor([[0.475], [0.69]]),
        verdict_logits=torch.randn(b, k_hyps),
        verdict_probs=torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]),
        tool_logits=torch.randn(b, n_tools),
        tool_gains=torch.tensor([[0.9, 0.1, 0.0, 0.4], [0.2, 0.3, 0.8, 0.0]]),
        stall=torch.tensor([False, True]),
        stall_mode="LEARNED",
        rule_stall=torch.tensor([False, False]),
        selected_tools=[["tool_a"], []],
        proposed_action=["CALL_TOOL", "STALL"],
        h_T=torch.randn(b, 1024),
        tool_names=["tool_a", "tool_b", "tool_c", "tool_d"],
        hyp_names=["H_stolen", "H_friendly", "H_novel"],
        conformal_set=[["H_stolen"], ["H_friendly", "H_novel"]],
        gain_head_trained=True,
        is_truncated=False,
    )

    trace = out.to_trace_dict(batch_idx=0)

    # 1. Check all values are primitive Python types (int, float, str, bool, list, dict)
    def assert_primitive(obj, path=""):
        if isinstance(obj, (int, float, str, bool)) or obj is None:
            return
        elif isinstance(obj, list):
            for i, it in enumerate(obj):
                assert_primitive(it, f"{path}[{i}]")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                assert isinstance(k, str), f"Key at {path} is not string: {k}"
                assert_primitive(v, f"{path}.{k}")
        else:
            pytest.fail(f"Non-primitive type {type(obj)} found at {path}: {obj}")

    assert_primitive(trace)

    # 2. Check semantic contents
    assert trace["s_prob"] == 0.7685
    assert trace["r_prob"] == 0.475
    assert trace["action"] == "CALL_TOOL"
    assert trace["selected_tools"] == ["tool_a"]
    assert trace["stall"] is False
    assert trace["stall_mode"] == "LEARNED"
    assert "tool_a" in trace["tool_gains"]
    assert "H_stolen" in trace["verdict_probs"]
    assert trace["conformal_set"] == ["H_stolen"]
