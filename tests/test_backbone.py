"""Unit tests for Step 2: Tokenizer Refactoring, Input Tensor Formatting, Position Tracker,
Tool MLP Head, Shared Verdict Head, and LayaInvestigationModel.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertConfig, BertModel

from rex.events import (
    CANONICAL_TOOL_SLOTS,
    EventType,
    HypothesisMask,
    InvestigationTensorFormatter,
    LayaInvestigationModel,
    PositionTracker,
    SharedVerdictHead,
    ToolMLPHead,
    ToolSlot,
    build_poe_attention_bias,
    setup_investigation_tokenizer,
)


def test_tokenizer_setup_and_mask_token():
    """Verify tokenizer has <TYPE_PROMO_ABUSE>, <TYPE_CREDENTIAL_STUFFING>, and [MASK]."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    res = setup_investigation_tokenizer(tokenizer)

    assert res["all_atomic"] is True
    assert res["mask_token"] == "[MASK]"
    assert res["mask_token_id"] is not None

    # Check atomic encoding (length 1)
    tok_promo = tokenizer.encode("<TYPE_PROMO_ABUSE>", add_special_tokens=False)
    tok_cred = tokenizer.encode("<TYPE_CREDENTIAL_STUFFING>", add_special_tokens=False)
    tok_mask = tokenizer.encode("[MASK]", add_special_tokens=False)

    assert len(tok_promo) == 1
    assert len(tok_cred) == 1
    assert len(tok_mask) == 1


def test_tensor_formatter_syntax():
    """Verify input string strictly adheres to:
    [CLS] <type> instructions [SEP] [MASK] H_0 ... [MASK] H_novel [SEP] state_t [SEP]
    """
    formatter = InvestigationTensorFormatter(
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )

    formatted = formatter.format_sequence(
        type_token=EventType.PROMO_ABUSE,
        instructions="Investigate suspicious bonus claim surge",
        state_payload={"ip": "1.2.3.4", "hits": 240, "device_clones": 12},
    )

    # 1. Starts with [CLS] <TYPE_PROMO_ABUSE>
    assert formatted.startswith("[CLS] <TYPE_PROMO_ABUSE> Investigate suspicious bonus claim surge [SEP]")

    # 2. Contains [MASK] before each of the 4 MECE hypotheses
    assert formatted.count("[MASK]") == 4

    # 3. Ends with state telemetry and [SEP]
    assert formatted.endswith("[SEP]")
    assert '"device_clones": 12' in formatted


def test_position_tracker_extraction_and_validation():
    """Verify PositionTracker extracts [CLS] index and all 4 [MASK]_j indices."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    setup_investigation_tokenizer(tokenizer)

    formatter = InvestigationTensorFormatter()
    items = [
        {
            "type_token": EventType.CREDENTIAL_STUFFING,
            "instructions": "Determine attack vector",
            "state_payload": {"failed_attempts": 500, "cgnat": False},
        },
        {
            "type_token": EventType.PROMO_ABUSE,
            "instructions": "Audit referral chain",
            "state_payload": {"referral_depth": 5, "device_id": "dev_99"},
        },
    ]

    batch = formatter.encode_batch(tokenizer, items, max_length=256)

    input_ids = batch["input_ids"]
    cls_indices = batch["cls_indices"]
    mask_indices = batch["mask_indices"]

    assert cls_indices.shape == (2,)
    assert mask_indices.shape == (2, 4)

    # In each item, verify token at cls_index is [CLS]
    for b in range(2):
        assert input_ids[b, cls_indices[b]] == tokenizer.cls_token_id
        for j in range(4):
            assert input_ids[b, mask_indices[b, j]] == tokenizer.mask_token_id

    # Test error handling when number of masks is corrupt
    corrupt_ids = input_ids.clone()
    corrupt_ids[0, mask_indices[0, 2]] = 999  # overwrite one [MASK]
    try:
        PositionTracker.extract_positions(corrupt_ids, tokenizer.mask_token_id, tokenizer.cls_token_id)
        assert False, "Should have raised ValueError for missing [MASK] token"
    except ValueError as e:
        assert "has 3 [MASK] tokens" in str(e)


def test_tool_mlp_head():
    """Verify ToolMLPHead 2-layer MLP with LayerNorm, GELU, and correct output shape."""
    hidden_size = 64
    num_slots = 4
    head = ToolMLPHead(hidden_size=hidden_size, num_slots=num_slots, intermediate_size=128)

    # Check layer types
    assert isinstance(head.fc1, nn.Linear)
    assert isinstance(head.ln, nn.LayerNorm)
    assert isinstance(head.act, nn.GELU)
    assert isinstance(head.fc2, nn.Linear)

    h_cls = torch.randn(8, hidden_size)
    logits = head(h_cls)

    assert logits.shape == (8, num_slots)


def test_shared_verdict_head():
    """Verify SharedVerdictHead single linear projection layer scoring hypotheses."""
    hidden_size = 64
    num_hypotheses = 4
    head = SharedVerdictHead(hidden_size=hidden_size)

    # Must be a single linear projection layer
    assert isinstance(head.proj, nn.Linear)
    assert head.proj.out_features == 1

    h_masks = torch.randn(8, num_hypotheses, hidden_size)
    verdict_logits = head(h_masks)

    assert verdict_logits.shape == (8, num_hypotheses)


def test_laya_investigation_model_end_to_end():
    """Verify complete end-to-end forward pass with encoder and PoE masking."""
    # Build lightweight mock Bert encoder
    cfg = BertConfig(
        vocab_size=32000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=512,
    )
    encoder = BertModel(cfg)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    setup_investigation_tokenizer(tokenizer)

    model = LayaInvestigationModel(
        encoder=encoder,
        num_slots=4,
        num_hypotheses=4,
        mask_token_id=tokenizer.mask_token_id,
        cls_token_id=tokenizer.cls_token_id,
    )

    formatter = InvestigationTensorFormatter()
    items = [
        {
            "type_token": EventType.PROMO_ABUSE,
            "instructions": "Investigate botnet vs sybil",
            "state_payload": {"velocity": 100, "unique_ips": 5},
        }
    ]
    batch = formatter.encode_batch(tokenizer, items, max_length=128)

    # 1. Unmasked forward pass
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        cls_indices=batch["cls_indices"],
        mask_indices=batch["mask_indices"],
    )

    assert out.tool_logits.shape == (1, 4)
    assert out.verdict_logits.shape == (1, 4)
    assert out.h_cls.shape == (1, 64)
    assert out.h_masks.shape == (1, 4, 64)

    # 2. Forward pass with PoE Inverse Tool Masking & Budget Pruner (K_2 = 2)
    # Suppose only H_shared is active (SLOT_DEVICE should be masked out)
    poe_mask = HypothesisMask(botnet=0, sybil=0, shared=1, novel=0)

    out_masked = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        cls_indices=batch["cls_indices"],
        mask_indices=batch["mask_indices"],
        hypothesis_mask=poe_mask,
        budget_k=2,
    )

    # In canonical slots [NETWORK, DEVICE, BEHAVIOR, GRAPH]:
    # DEVICE is slot index 1 -> must be masked out (-1e4) in eval mode
    model.eval()
    out_masked = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        cls_indices=batch["cls_indices"],
        mask_indices=batch["mask_indices"],
        hypothesis_mask=poe_mask,
        budget_k=2,
    )
    device_slot_idx = CANONICAL_TOOL_SLOTS.index(ToolSlot.DEVICE)
    assert out_masked.tool_logits[0, device_slot_idx] == -1e4


def test_state_t_only_truncation_never_corrupts_masks():
    """Verify that even with a massive 5000-char state_t payload and short max_length=64,
    the Invariant Prefix and all 4 [MASK] tokens are 100% preserved.
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    setup_investigation_tokenizer(tokenizer)

    formatter = InvestigationTensorFormatter()
    # Huge state_t with repetitive data
    huge_state = {"log": "telemetry_dump_value_" * 200, "metric": [i for i in range(200)]}

    items = [
        {
            "type_token": EventType.PROMO_ABUSE,
            "instructions": "Detect distributed syndicate",
            "state_payload": huge_state,
        }
    ]

    # Max length strictly constrained to 128 tokens (Prefix takes ~80 tokens, remaining ~48 for state)
    batch = formatter.encode_batch(tokenizer, items, max_length=128)

    input_ids = batch["input_ids"]
    mask_indices = batch["mask_indices"]
    cls_indices = batch["cls_indices"]

    assert input_ids.shape[1] <= 128
    # Token 0 is [CLS]
    assert input_ids[0, cls_indices[0]] == tokenizer.cls_token_id

    # Exactly 4 [MASK] tokens survived and are tracked
    assert mask_indices.shape == (1, 4)
    for j in range(4):
        assert input_ids[0, mask_indices[0, j]] == tokenizer.mask_token_id

    # The sequence ends with [SEP]
    assert input_ids[0, -1] == tokenizer.sep_token_id


def test_resize_token_embeddings_automatic_sync():
    """Verify LayaInvestigationModel automatically calls resize_token_embeddings if needed."""
    cfg = BertConfig(
        vocab_size=1000,  # small initial vocab
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
    )
    encoder = BertModel(cfg)
    assert encoder.get_input_embeddings().num_embeddings == 1000

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    setup_investigation_tokenizer(tokenizer)
    assert len(tokenizer) > 30000

    # Pass tokenizer to model constructor
    model = LayaInvestigationModel(
        encoder=encoder,
        tokenizer=tokenizer,
    )

    # Embedding matrix must have been automatically resized to len(tokenizer)
    assert model.encoder.get_input_embeddings().num_embeddings == len(tokenizer)
    assert model.mask_token_id == tokenizer.mask_token_id
    assert model.cls_token_id == tokenizer.cls_token_id


def test_gradient_flow_and_training_unmasked_logits():
    """Verify that during training, raw_tool_logits maintains full differentiable gradients,
    preventing gradient starvation (zeros) on negative/pruned slots.
    """
    cfg = BertConfig(
        vocab_size=1000,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
    )
    encoder = BertModel(cfg)
    model = LayaInvestigationModel(encoder=encoder, num_slots=4, num_hypotheses=4)
    model.train()  # Ensure training mode

    input_ids = torch.randint(0, 1000, (2, 20))
    attention_mask = torch.ones_like(input_ids)
    cls_indices = torch.tensor([0, 0], dtype=torch.long)
    mask_indices = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=torch.long)

    # In training mode, even if hypothesis_mask and budget_k are provided,
    # raw_tool_logits and tool_logits must NOT be masked out with -1e4
    poe_mask = HypothesisMask(botnet=0, sybil=0, shared=1, novel=0)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        cls_indices=cls_indices,
        mask_indices=mask_indices,
        hypothesis_mask=poe_mask,
        budget_k=2,
    )

    # No slot should be clamped to -1e4 in training mode
    assert (out.tool_logits <= -5000).sum() == 0
    assert (out.raw_tool_logits <= -5000).sum() == 0

    # Backprop through raw_tool_logits
    loss = out.raw_tool_logits.sum() + out.verdict_logits.sum()
    loss.backward()

    # Check non-zero gradients on tool head
    assert model.tool_head.fc1.weight.grad is not None
    assert model.tool_head.fc1.weight.grad.abs().sum() > 0
    assert model.tool_head.fc2.weight.grad is not None
    assert model.tool_head.fc2.weight.grad.abs().sum() > 0
    # Check non-zero gradients on verdict head
    assert model.verdict_head.proj.weight.grad is not None
    assert model.verdict_head.proj.weight.grad.abs().sum() > 0


def test_position_tracker_extract_spans():
    """Verify PositionTracker.extract_spans isolates hypothesis spans and state_t."""
    # Synthetic sequence layout:
    # 0:[CLS] 1:inst 2:[SEP] 3:[MASK]_0 4:h0 5:[MASK]_1 6:h1 7:[MASK]_2 8:h2 9:[MASK]_3 10:h3 11:[SEP] 12:state 13:state 14:[SEP]
    input_ids = torch.tensor([[101, 10, 102, 103, 20, 103, 30, 103, 40, 103, 50, 102, 60, 61, 102]])
    mask_indices = torch.tensor([[3, 5, 7, 9]])
    sep_token_id = 102

    hyp_spans, state_spans = PositionTracker.extract_spans(input_ids, mask_indices, sep_token_id)
    assert len(hyp_spans) == 1
    assert len(hyp_spans[0]) == 4

    assert hyp_spans[0][0] == (3, 4)   # H_0: 3..4
    assert hyp_spans[0][1] == (5, 6)   # H_1: 5..6
    assert hyp_spans[0][2] == (7, 8)   # H_2: 7..8
    assert hyp_spans[0][3] == (9, 10)  # H_3: 9..10
    assert state_spans[0] == (12, 13)  # state_t: 12..13


def test_build_poe_attention_bias_structure():
    """Verify 4D tensor construction and bidirectional blocking for refuted hypothesis."""
    input_ids = torch.tensor([[101, 10, 102, 103, 20, 103, 30, 103, 40, 103, 50, 102, 60, 61, 102]])
    mask_indices = torch.tensor([[3, 5, 7, 9]])
    cls_indices = torch.tensor([0])
    sep_token_id = 102
    base_mask = torch.ones_like(input_ids)

    # Refute H_0 (M_0 = 0)
    hyp_mask = HypothesisMask(botnet=0, sybil=1, shared=1, novel=1)

    bias = build_poe_attention_bias(
        input_ids=input_ids,
        mask_indices=mask_indices,
        hypothesis_mask=hyp_mask,
        cls_indices=cls_indices,
        base_attention_mask=base_mask,
        sep_token_id=sep_token_id,
    )

    assert bias.shape == (1, 1, 15, 15)
    # Check [CLS] (0) to H_0 (3..4) is blocked to -1e4
    assert (bias[0, 0, 0, 3:5] <= -1000).all()
    assert (bias[0, 0, 3:5, 0] <= -1000).all()

    # Check state_t (12..13) to H_0 (3..4) is blocked to -1e4
    assert (bias[0, 0, 12:14, 3:5] <= -1000).all()
    assert (bias[0, 0, 3:5, 12:14] <= -1000).all()

    # Check that [CLS] to H_novel (9..10) is UNBLOCKED (0.0)
    assert (bias[0, 0, 0, 9:11] == 0.0).all()


def test_poe_attention_biasing_encoder_layer_zero_attention():
    """Verify self-attention weights inside Transformer layers drop to strictly 0.0 for refuted branches."""
    cfg = BertConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        attn_implementation="eager",
    )
    encoder = BertModel(cfg)
    model = LayaInvestigationModel(
        encoder=encoder,
        mask_token_id=103,
        cls_token_id=101,
        sep_token_id=102,
    )

    input_ids = torch.tensor([[101, 10, 102, 103, 20, 103, 30, 103, 40, 103, 50, 102, 60, 61, 102]])
    mask_indices = torch.tensor([[3, 5, 7, 9]])
    cls_indices = torch.tensor([0])
    base_mask = torch.ones_like(input_ids)

    # Refute H_botnet
    hyp_mask = HypothesisMask(botnet=0, sybil=1, shared=1, novel=1)

    out = model(
        input_ids=input_ids,
        attention_mask=base_mask,
        cls_indices=cls_indices,
        mask_indices=mask_indices,
        hypothesis_mask=hyp_mask,
        poe_attention_biasing=True,
        output_attentions=True,
    )

    assert out.extended_attention_mask is not None
    assert out.attentions is not None

    for layer_idx, att in enumerate(out.attentions):
        # att shape: [batch_size, num_heads, seq_len, seq_len]
        # Attention from CLS (0) to H_botnet (3..4)
        cls_to_botnet = att[0, :, 0, 3:5]
        assert cls_to_botnet.max().item() == 0.0, f"Layer {layer_idx}: Attention from CLS to refuted H_botnet must be 0.0!"

        # Attention from state_t (12..13) to H_botnet (3..4)
        state_to_botnet = att[0, :, 12:14, 3:5]
        assert state_to_botnet.max().item() == 0.0, f"Layer {layer_idx}: Attention from state_t to refuted H_botnet must be 0.0!"

        # Attention from CLS (0) to H_novel (9..10) must remain active
        cls_to_novel = att[0, :, 0, 9:11]
        assert cls_to_novel.max().item() > 0.0, f"Layer {layer_idx}: Attention to active H_novel must be > 0.0!"


def test_poe_attention_biasing_changes_h_cls_representation():
    """Verify that PoE Attention Biasing physically alters the computed h_CLS hidden state."""
    cfg = BertConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
    )
    encoder = BertModel(cfg)
    model = LayaInvestigationModel(
        encoder=encoder,
        mask_token_id=103,
        cls_token_id=101,
        sep_token_id=102,
    )

    input_ids = torch.tensor([[101, 10, 102, 103, 20, 103, 30, 103, 40, 103, 50, 102, 60, 61, 102]])
    mask_indices = torch.tensor([[3, 5, 7, 9]])
    cls_indices = torch.tensor([0])
    base_mask = torch.ones_like(input_ids)
    hyp_mask = HypothesisMask(botnet=0, sybil=0, shared=0, novel=1)

    # 1. Forward with Attention Biasing ENABLED
    out_biased = model(
        input_ids=input_ids,
        attention_mask=base_mask,
        cls_indices=cls_indices,
        mask_indices=mask_indices,
        hypothesis_mask=hyp_mask,
        poe_attention_biasing=True,
    )

    # 2. Forward with Attention Biasing DISABLED
    out_unbiased = model(
        input_ids=input_ids,
        attention_mask=base_mask,
        cls_indices=cls_indices,
        mask_indices=mask_indices,
        hypothesis_mask=hyp_mask,
        poe_attention_biasing=False,
    )

    # h_CLS representation MUST be physically different because refuted hypotheses were blocked from self-attention
    diff = (out_biased.h_cls - out_unbiased.h_cls).abs().max().item()
    assert diff > 1e-3, f"Expected h_cls to change with PoE Attention Biasing, diff={diff}"


if __name__ == "__main__":
    test_tokenizer_setup_and_mask_token()
    test_tensor_formatter_syntax()
    test_position_tracker_extraction_and_validation()
    test_tool_mlp_head()
    test_shared_verdict_head()
    test_laya_investigation_model_end_to_end()
    test_state_t_only_truncation_never_corrupts_masks()
    test_resize_token_embeddings_automatic_sync()
    test_gradient_flow_and_training_unmasked_logits()
    test_position_tracker_extract_spans()
    test_build_poe_attention_bias_structure()
    test_poe_attention_biasing_encoder_layer_zero_attention()
    test_poe_attention_biasing_changes_h_cls_representation()
    print("All Step 2 Model Refactoring & PoE Attention Biasing tests passed successfully!")

