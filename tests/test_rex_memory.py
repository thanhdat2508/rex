"""Unit tests for incremental evidence encoding, 64-token cap, and memory manager (Step A4)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from rex.rex_format import encode_evidence
from rex.rex_memory import EvidenceMemoryManager
from rex.rex_ponder import LatentPonderBlock
from rex.rex_types import RexState


class DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [int(abs(hash(w)) % 1000) + 1 for w in text.split()]


def test_encode_evidence_token_cap():
    """Verify evidence text exceeding 64 tokens is strictly capped with is_truncated=True."""
    tok = DummyTokenizer()

    # Case 1: Short observation (10 tokens)
    short_text = "ip_country=SG device_fingerprint_match=true status=clean risk_score=2"
    enc_short = encode_evidence(tok, short_text, round_idx=2, max_tokens=64)
    assert len(enc_short["input_ids"]) <= 64
    assert not enc_short["is_truncated"]

    # Case 2: Long observation (100 tokens)
    long_text = " ".join([f"metric_{i}={i * 1.5}" for i in range(100)])
    enc_long = encode_evidence(tok, long_text, round_idx=2, max_tokens=64)
    assert len(enc_long["input_ids"]) == 64
    assert enc_long["is_truncated"]


def test_evidence_memory_manager_append_and_kv_extension():
    """Verify appending evidence expands M and extends KV-cache while preserving thought state h."""
    b, d = 2, 64
    state = RexState.empty(batch_size=b, d=d)
    initial_h = torch.randn(b, d)
    state.h = initial_h.clone()

    ponder_block = LatentPonderBlock(d_model=d, nhead=2)

    # 1. Turn 1 initial memory: L=12
    M_0 = torch.randn(b, 12, d)
    mask_0 = torch.ones(b, 12, dtype=torch.bool)
    state = EvidenceMemoryManager.append_evidence(state, M_0, mask_0, ponder_block=ponder_block)

    assert state.M.shape == (b, 12, d)
    assert state.memory_mask.shape == (b, 12)
    assert state.kv_cache[0].shape == (b, 2, 12, 32)
    # Thought state h must remain untouched
    torch.testing.assert_close(state.h, initial_h)

    # 2. Turn 2 incremental evidence: L=6
    M_new = torch.randn(b, 6, d)
    mask_new = torch.ones(b, 6, dtype=torch.bool)
    state = EvidenceMemoryManager.append_evidence(state, M_new, mask_new, is_truncated=True, ponder_block=ponder_block)

    assert state.M.shape == (b, 18, d)
    assert state.memory_mask.shape == (b, 18)
    assert state.kv_cache[0].shape == (b, 2, 18, 32)
    assert state.is_truncated is True
    torch.testing.assert_close(state.h, initial_h)
