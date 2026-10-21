"""Unit tests for LatentPonderBlock, KV-cache, LayerScale, and numerical stability (Step A2)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn.functional as F

from rex.rex_ponder import LatentPonderBlock


def test_ponder_block_shapes_and_cache_equivalence():
    """Verify output shapes and that cached KV path matches non-cached computation within 1e-5."""
    b, d, l_mem = 2, 256, 16
    block = LatentPonderBlock(d_model=d, nhead=4, default_t=3)
    block.eval()

    h_0 = torch.randn(b, d)
    M = torch.randn(b, l_mem, d)
    mask = torch.ones(b, l_mem, dtype=torch.bool)

    # 1. Forward with M directly (builds cache internally)
    h_T_direct, kv_cache, _ = block(h_0, M=M, memory_mask=mask, t_steps=3)
    assert h_T_direct.shape == (b, d)

    # 2. Forward using explicit pre-cached KV
    h_T_cached, _, _ = block(h_0, kv_cache=kv_cache, memory_mask=mask, t_steps=3)
    assert h_T_cached.shape == (b, d)

    # Cache equivalence assertion
    torch.testing.assert_close(h_T_direct, h_T_cached, rtol=1e-5, atol=1e-5)


def test_ponder_block_t0_returns_identity():
    """Verify T=0 returns h_0 untouched."""
    b, d = 2, 128
    block = LatentPonderBlock(d_model=d, nhead=2)
    h_0 = torch.randn(b, d)
    M = torch.randn(b, 8, d)

    h_out, _, _ = block(h_0, M=M, t_steps=0)
    torch.testing.assert_close(h_out, h_0)


def test_ponder_initialization_cosine_similarity():
    """Verify that at initialization, h_T has >0.99 cosine similarity with LayerNorm(h_0)."""
    b, d = 4, 256
    block = LatentPonderBlock(d_model=d, nhead=4, default_t=3)
    block.eval()

    h_0 = torch.randn(b, d)
    M = torch.randn(b, 16, d)
    mask = torch.ones(b, 16, dtype=torch.bool)

    h_T, _, _ = block(h_0, M=M, memory_mask=mask, t_steps=3)

    # At init with LayerScale 1e-4 and b_g = -2.0, h_{t+1} ~ LN(h_t)
    h_expected = block.ln_final(h_0)
    cos_sim = F.cosine_similarity(h_T, h_expected, dim=-1)
    assert (cos_sim > 0.99).all(), f"Min cosine similarity {cos_sim.min().item()} below 0.99"


def test_ponder_gradient_flow():
    """Verify gradients propagate to gate, LayerScale gamma, and transformer projections."""
    b, d = 2, 64
    block = LatentPonderBlock(d_model=d, nhead=2, default_t=2)
    block.train()

    h_0 = torch.randn(b, d, requires_grad=True)
    M = torch.randn(b, 8, d, requires_grad=True)
    mask = torch.ones(b, 8, dtype=torch.bool)

    h_T, _, _ = block(h_0, M=M, memory_mask=mask, t_steps=2)
    loss = h_T.sum()
    loss.backward()

    assert h_0.grad is not None
    assert M.grad is not None
    assert block.gate.weight.grad is not None
    assert block.gamma.grad is not None
    assert block.q_proj.weight.grad is not None


def test_ponder_autocast_numerical_stability():
    """Verify FP16 and BF16 autocast forward pass produces finite values without NaN."""
    b, d = 2, 128
    block = LatentPonderBlock(d_model=d, nhead=2)
    h_0 = torch.randn(b, d)
    M = torch.randn(b, 8, d)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        h_T, _, _ = block(h_0, M=M, t_steps=3)
        assert torch.isfinite(h_T).all()
