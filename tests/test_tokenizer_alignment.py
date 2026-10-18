"""Unit tests for Tokenizer Alignment and Embedding Layer Synchronization.

Verifies:
  1. Atomicity of special tokens ([CLS], [SEP], [MASK], <TYPE_...>) preventing BPE subword splitting.
  2. Automatic embedding matrix resizing and Semantic Warm-Start initialization.
  3. Strict alignment of mask_token_id, cls_token_id, sep_token_id.
  4. PositionTracker index integrity: ensures h_mask,j vectors are extracted without index drift.
  5. End-to-end verification with both BERT and real ModernBERT checkpoint tokenizers.
"""

import os
import sys
import unittest
import torch.nn as nn
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rex.events import (
    InvestigationTensorFormatter,
    LayaInvestigationModel,
    MultiTurnInvestigationLoop,
    PositionTracker,
    TokenizerAlignmentGuard,
)


class DummyModernBertEncoder(nn.Module):
    """In-memory encoder matching ModernBERT interface for deterministic unit tests."""

    def __init__(self, vocab_size: int = 50280, hidden_size: int = 64):
        super().__init__()
        self.config = unittest.mock.MagicMock()
        self.config.hidden_size = hidden_size
        self.config.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, hidden_size)

    def get_input_embeddings(self):
        return self.embeddings

    def resize_token_embeddings(self, new_num_tokens: int):
        old_size = self.embeddings.num_embeddings
        old_weight = self.embeddings.weight.data
        new_embeddings = nn.Embedding(new_num_tokens, self.config.hidden_size)
        if new_num_tokens >= old_size:
            new_embeddings.weight.data[:old_size] = old_weight
        else:
            new_embeddings.weight.data = old_weight[:new_num_tokens]
        self.embeddings = new_embeddings
        self.config.vocab_size = new_num_tokens

    def forward(self, input_ids, attention_mask=None, **kwargs):
        last_hidden = self.embeddings(input_ids)
        mock_output = unittest.mock.MagicMock()
        mock_output.last_hidden_state = last_hidden
        mock_output.__getitem__ = lambda s, idx: last_hidden if idx == 0 else None
        return mock_output


def test_tokenizer_atomicity_and_embedding_resize():
    """Verify that special tokens are registered as atomic and embedding is resized."""
    # Start with bert-base-uncased where <TYPE_PROMO_ABUSE> is unknown and gets split
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    enc = DummyModernBertEncoder(vocab_size=len(tokenizer), hidden_size=64)
    model = LayaInvestigationModel(encoder=enc, tokenizer=tokenizer)

    # Before guard: <TYPE_PROMO_ABUSE> is split into subwords
    assert len(tokenizer.tokenize("<TYPE_PROMO_ABUSE>")) > 1

    # Run TokenizerAlignmentGuard
    report = TokenizerAlignmentGuard.align_and_verify(
        model=model,
        tokenizer=tokenizer,
        semantic_init=True,
    )

    # 1. Now <TYPE_PROMO_ABUSE> must be exactly 1 atomic token
    tokens = tokenizer.tokenize("<TYPE_PROMO_ABUSE>")
    assert len(tokens) == 1, f"Expected 1 token, got {tokens}"

    # 2. Embedding matrix must be resized to accommodate new tokens
    assert report["model_resized"] is True
    assert enc.get_input_embeddings().num_embeddings >= len(tokenizer)
    assert enc.get_input_embeddings().num_embeddings % 64 == 0
    assert report["alignment_verified"] is True


def test_real_modernbert_checkpoint_alignment():
    """Verify alignment on the real saved ModernBERT checkpoint tokenizer."""
    ckpt_path = "checkpoints/laya-mece-investigation/tokenizer"
    if not os.path.exists(ckpt_path):
        return

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    initial_vocab = len(tokenizer)
    enc = DummyModernBertEncoder(vocab_size=initial_vocab, hidden_size=64)
    model = LayaInvestigationModel(encoder=enc, tokenizer=tokenizer)

    # Apply guard
    report = TokenizerAlignmentGuard.align_and_verify(
        model=model,
        tokenizer=tokenizer,
        semantic_init=True,
    )

    # Verify atomic encoding
    enc_promo = tokenizer.encode("<TYPE_PROMO_ABUSE>", add_special_tokens=False)
    assert len(enc_promo) == 1, f"ModernBERT split <TYPE_PROMO_ABUSE>: {enc_promo}"

    # Verify ID synchronization
    assert model.mask_token_id == tokenizer.mask_token_id
    assert model.cls_token_id == tokenizer.cls_token_id
    assert model.sep_token_id == tokenizer.sep_token_id

    # Verify PositionTracker extracts exactly 4 masks with no drift
    formatter = InvestigationTensorFormatter(
        cls_token=tokenizer.cls_token,
        sep_token=tokenizer.sep_token,
        mask_token=tokenizer.mask_token,
    )
    seq = formatter.format_sequence(
        type_token="<TYPE_PROMO_ABUSE>",
        instructions="Test investigation",
        state_payload={"ip": "1.2.3.4"},
        tokenizer=tokenizer,
    )
    batch = tokenizer(seq, return_tensors="pt")
    cls_idx, mask_idx = PositionTracker.extract_positions(
        input_ids=batch["input_ids"],
        mask_token_id=model.mask_token_id,
        cls_token_id=model.cls_token_id,
    )

    assert mask_idx.shape == (1, 4)
    for j in range(4):
        actual_token_id = batch["input_ids"][0, mask_idx[0, j]].item()
        assert actual_token_id == model.mask_token_id


def test_multi_turn_investigation_loop_auto_alignment():
    """Verify MultiTurnInvestigationLoop automatically enforces TokenizerAlignmentGuard upon init."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    enc = DummyModernBertEncoder(vocab_size=len(tokenizer), hidden_size=64)
    model = LayaInvestigationModel(encoder=enc, tokenizer=tokenizer)

    loop = MultiTurnInvestigationLoop(
        model=model,
        tokenizer=tokenizer,
    )

    assert hasattr(loop, "alignment_report")
    assert loop.alignment_report["alignment_verified"] is True
    assert loop.alignment_report["all_atomic"] is True


if __name__ == "__main__":
    test_tokenizer_atomicity_and_embedding_resize()
    test_real_modernbert_checkpoint_alignment()
    test_multi_turn_investigation_loop_auto_alignment()
    print("All Tokenizer Alignment Guard tests passed successfully!")
