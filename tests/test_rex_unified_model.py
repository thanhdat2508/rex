"""Unit tests for Rex unified model (arch='rex2'), 4 decision heads, and checkpoint parity (Step A8)."""
import os
import shutil
import tempfile
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from rex.common import DecisionModel
from rex.rex_heads import VerdictBridge
from rex.rex_model import RexDecisionModel, from_rex, rex_config, save_rex_checkpoint
from rex.rex_registry import ToolSpec
from rex.rex_types import RexState, RexStepOutput


class MockEncoder(nn.Module):
    """Lightweight deterministic encoder for test suite."""

    def __init__(self, hidden_size: int = 32, vocab_size: int = 100):
        super().__init__()
        self.config = PretrainedConfig()
        self.config.hidden_size = hidden_size
        self.config.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        class Output:
            pass

        out = Output()
        out.last_hidden_state = self.embeddings(input_ids)
        return out


def _create_mock_rex_model(d: int = 32, n_tools: int = 3) -> RexDecisionModel:
    enc = MockEncoder(hidden_size=d)
    tools = [f"tool_{i}" for i in range(n_tools)]
    return RexDecisionModel(
        encoder=enc,
        tools=tools,
        head_layers=1,
        n_act=2,
        ponder_steps=2,
    )


def test_rex_unified_model_init():
    """Verify that RexDecisionModel initializes all 4 heads, ponder block, and bridge."""
    model = _create_mock_rex_model(d=32, n_tools=3)

    assert hasattr(model, "ponder_block")
    assert hasattr(model, "head_s")
    assert hasattr(model, "head_r")
    assert hasattr(model, "verdict_bridge")
    assert hasattr(model, "tool_matcher")
    assert hasattr(model, "tool_gain")
    assert hasattr(model, "hyp_scorer")
    assert hasattr(model, "tool_embeddings")
    assert model.tool_embeddings.shape == (3, 32)
    assert model.tool_head is model.tool_matcher


def test_rex_unified_model_forward_turn1():
    """Verify Turn 1 forward execution with initial state construction."""
    model = _create_mock_rex_model(d=32, n_tools=3)
    B, S, K = 2, 8, 4

    input_ids = torch.randint(0, 50, (B, S))
    attention_mask = torch.ones((B, S), dtype=torch.long)
    marker_pos = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=torch.long)
    marker_mask = torch.ones((B, K), dtype=torch.bool)

    tool_specs = [
        ToolSpec(name="tool_0", returns=[], level="READ", cost=0.1),
        ToolSpec(name="tool_1", returns=[], level="LIGHT", cost=0.4),
        ToolSpec(name="tool_2", returns=[], level="HEAVY", cost=0.8),
    ]

    verdict_logits, state, step_out = model.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        marker_pos=marker_pos,
        marker_mask=marker_mask,
        tool_specs=tool_specs,
    )

    assert isinstance(state, RexState)
    assert isinstance(step_out, RexStepOutput)
    assert verdict_logits.shape == (B, K)
    assert state.round_idx == 1
    assert state.M.shape == (B, S, 32)
    assert state.kv_cache is not None
    assert step_out.s_prob.shape == (B, 1)
    assert step_out.r_prob.shape == (B, 1)
    assert step_out.tool_logits.shape == (B, 3)
    assert step_out.tool_gains.shape == (B, 3)
    assert len(step_out.selected_tools) == B
    assert len(step_out.proposed_action) == B


def test_rex_unified_model_forward_turn2_incremental():
    """Verify Turn 2 forward execution with incremental evidence appending."""
    model = _create_mock_rex_model(d=32, n_tools=3)
    B, S, K = 2, 8, 4

    input_ids = torch.randint(0, 50, (B, S))
    attention_mask = torch.ones((B, S), dtype=torch.long)
    marker_pos = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=torch.long)
    marker_mask = torch.ones((B, K), dtype=torch.bool)

    # Turn 1
    _, state1, _ = model.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        marker_pos=marker_pos,
        marker_mask=marker_mask,
    )
    assert state1.round_idx == 1
    initial_mem_len = state1.M.size(1)

    # Turn 2: Provide new evidence
    evidence_new = torch.randint(0, 50, (B, 4))
    evidence_mask = torch.ones((B, 4), dtype=torch.bool)

    verdict_logits2, state2, step_out2 = model.forward(
        state=state1,
        evidence_new=evidence_new,
        evidence_mask=evidence_mask,
        marker_mask=marker_mask,
    )

    assert state2.round_idx == 2
    assert state2.M.size(1) == initial_mem_len + 4
    assert verdict_logits2.shape == (B, K)
    assert step_out2.h_T.shape == (B, 32)


def test_zero_divergence_laya_checkpoint_parity():
    """Verify that from_rex produces mathematically zero divergence on verdict logits."""
    d = 32
    enc = MockEncoder(hidden_size=d)
    laya_model = DecisionModel(enc, head_layers=1, n_act=2)

    tools = ["disk_read", "net_ping"]
    rex_model = from_rex(laya_model, tools=tools)

    assert isinstance(rex_model, RexDecisionModel)

    laya_model.eval()
    rex_model.eval()

    # Prepare inputs
    B, S, K = 2, 6, 3
    input_ids = torch.randint(0, 50, (B, S))
    attention_mask = torch.ones((B, S), dtype=torch.long)
    marker_pos = torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.long)
    marker_mask = torch.ones((B, K), dtype=torch.bool)
    qtype = torch.zeros(B, dtype=torch.long)

    # Laya decide output
    with torch.no_grad():
        h_laya = laya_model._contextualize(input_ids, attention_mask, qtype)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, d)
        m_laya = torch.gather(h_laya, 1, idx)
        verdict_laya = laya_model.scorer(m_laya).squeeze(-1).float()
        verdict_laya = verdict_laya.masked_fill(~marker_mask, -1e4)

    # Rex forward output via rex_forward
    with torch.no_grad():
        out_rex = rex_model.rex_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            marker_pos=marker_pos,
            marker_mask=marker_mask,
            qtype=qtype,
        )
        verdict_rex = out_rex["verdict_logits"]

    # Bridge weights are zero (W_v=0, g_v=0) -> output must be identical
    max_delta = (verdict_laya - verdict_rex).abs().max().item()
    assert max_delta < 1e-6, f"Verdict logit delta {max_delta} exceeds 1e-6 tolerance."


def test_rex_config_arch2():
    """Verify that rex_config assigns arch='rex2'."""
    model = _create_mock_rex_model(d=32, n_tools=2)
    cfg = rex_config(model)
    assert cfg["arch"] == "rex2"
    assert cfg["rex"]["tools"] == ["tool_0", "tool_1"]
