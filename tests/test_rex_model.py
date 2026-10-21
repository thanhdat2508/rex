"""Comprehensive offline unit tests for Rex Architecture & Tokenizer (Part 2).

Tests DecisionModel refactoring, ToolHead, RexDecisionModel, rex_vocab, rex_format,
PoE elimination masking, and from_rex checkpoint conversion.
Runs completely offline using dummy in-memory Bert configs without network calls.
"""
import math
import os
import sys
import tempfile

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, BertTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rex.common import DecisionModel
from rex.rex_format import collate_rex, encode_rex_input
from rex.rex_model import RexDecisionModel, ToolHead, from_rex, rex_config, save_rex_checkpoint
from rex.rex_tools import ToolMatcherHead
from rex.rex_vocab import add_type_tokens, check_type_token


def _get_offline_tok():
    """Create a fast offline BertTokenizer with basic tokens."""
    fd, vocab_path = tempfile.mkstemp(suffix=".txt", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(
            "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n"
            "which\nhypothesis\nexplains\nthe\nsignup\nburst\n"
            "promotion\nabuse\nbotnet\nsybil\nshared\nnovel\n"
            "state\nevent\nlatency\nerror\n"
        )
    tok = BertTokenizer(vocab_path)
    os.remove(vocab_path)
    return tok


def _tiny_encoder(hidden_size: int = 32, vocab_size: int = 50) -> nn.Module:
    cfg = AutoConfig.for_model(
        "bert",
        hidden_size=hidden_size,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        vocab_size=vocab_size,
    )
    return AutoModel.from_config(cfg)


def _dummy_inputs(batch: int = 2, seq: int = 16, n_markers: int = 4, d: int = 32, vocab_size: int = 50):
    input_ids = torch.randint(0, vocab_size, (batch, seq))
    attention_mask = torch.ones(batch, seq, dtype=torch.long)
    qtype = torch.zeros(batch, dtype=torch.long)
    marker_pos = torch.arange(n_markers).unsqueeze(0).expand(batch, -1).clone()
    marker_mask = torch.ones(batch, n_markers, dtype=torch.bool)
    return input_ids, attention_mask, marker_pos, marker_mask, qtype


# ==============================================================================
# 1. DecisionModel Refactoring Tests (Step 2.0)
# ==============================================================================

def test_contextualize_and_decide_equivalence():
    """_contextualize + _decide must produce bit-for-bit identical results to forward()."""
    torch.manual_seed(42)
    enc = _tiny_encoder()
    model = DecisionModel(enc, head_layers=1, n_act=2)
    model.eval()

    input_ids, attention_mask, marker_pos, marker_mask, qtype = _dummy_inputs()

    with torch.no_grad():
        # Unified forward
        fwd_logits, fwd_act = model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
        # Split methods
        h = model._contextualize(input_ids, attention_mask, qtype)
        split_logits, split_act = model._decide(h, attention_mask, marker_pos, marker_mask)

    assert torch.allclose(fwd_logits, split_logits, atol=1e-7)
    assert torch.allclose(fwd_act, split_act, atol=1e-7)


def test_contextualize_detach_encoder():
    """detach_encoder=True detaches representation from encoder parameters."""
    enc = _tiny_encoder()
    model = DecisionModel(enc, head_layers=1, n_act=2)
    model.train()

    input_ids, attention_mask, marker_pos, marker_mask, qtype = _dummy_inputs()
    h = model._contextualize(input_ids, attention_mask, qtype, detach_encoder=True)
    loss = h.sum()
    loss.backward()

    # Encoder parameters should have no gradients
    for p in model.encoder.parameters():
        assert p.grad is None


# ==============================================================================
# 2. ToolHead Tests (Step 2.3)
# ==============================================================================

def test_tool_head_shapes_and_dtype():
    """ToolMatcherHead returns float32 logits of shape [B, N_tools]."""
    d, n_tools = 32, 6
    head = ToolMatcherHead(d=d, proj_dim=64)
    head.set_tool_embeddings(torch.randn(n_tools, d))
    x = torch.randn(4, d)
    logits = head(x)

    assert logits.shape == (4, n_tools)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_tool_head_tool_embeddings_property():
    """tool_embeddings and weight properties expose [n_tools, proj_dim] matrix for ortho loss."""
    d, n_tools, hidden = 32, 5, 48
    head = ToolMatcherHead(d=d, proj_dim=hidden)
    head.set_tool_embeddings(torch.randn(n_tools, d))

    assert head.tool_projection_weight.shape == (n_tools, hidden)
    assert head.tool_embeddings.shape == (n_tools, hidden)


# ==============================================================================
# 3. RexDecisionModel & PoE Forward Tests (Step 2.3 & 2.4)
# ==============================================================================

def test_rex_decision_model_forward():
    """RexDecisionModel forward() returns (verdict_logits, state, step_output)."""
    enc = _tiny_encoder()
    tools = ["tool_a", "tool_b", "tool_c"]
    model = RexDecisionModel(enc, tools=tools, head_layers=1, n_act=2)
    model.eval()

    input_ids, attention_mask, marker_pos, marker_mask, qtype = _dummy_inputs()
    with torch.no_grad():
        verdict_logits, state, step_out = model(input_ids, attention_mask, marker_pos, marker_mask, qtype)

    assert verdict_logits.shape == (2, 4)
    assert state.round_idx == 1
    assert step_out.verdict_logits.shape == (2, 4)


def test_rex_forward_outputs():
    """rex_forward returns dictionary with verdict_logits, tool_logits, act_logits, and h_cls."""
    enc = _tiny_encoder()
    tools = ["ip_lookup", "dump_traces", "kill_pod"]
    model = RexDecisionModel(enc, tools=tools, head_layers=1, n_act=2)
    model.eval()

    input_ids, attention_mask, marker_pos, marker_mask, qtype = _dummy_inputs(n_markers=3)
    with torch.no_grad():
        out = model.rex_forward(input_ids, attention_mask, marker_pos, marker_mask)

    assert set(out.keys()) == {"verdict_logits", "tool_logits", "act_logits", "h_cls"}
    assert out["verdict_logits"].shape == (2, 3)
    assert out["tool_logits"].shape == (2, len(tools))
    assert out["act_logits"].shape == (2, 2)
    assert out["h_cls"].shape == (2, 32)


def test_poe_elimination_hard_masking():
    """Eliminated hypotheses receive logit -1e4 and probability ~0."""
    enc = _tiny_encoder()
    model = RexDecisionModel(enc, tools=["tool_a"], head_layers=1, n_act=2)
    model.eval()

    input_ids, attention_mask, marker_pos, marker_mask, _ = _dummy_inputs(batch=1, n_markers=4)
    # Eliminate hypotheses at index 0 and 2
    hyp_alive = torch.tensor([[False, True, False, True]], dtype=torch.bool)

    with torch.no_grad():
        out = model.rex_forward(input_ids, attention_mask, marker_pos, marker_mask, hyp_alive=hyp_alive)
        logits = out["verdict_logits"][0]
        probs = torch.softmax(logits, dim=-1)

    assert logits[0].item() == -1e4
    assert logits[2].item() == -1e4
    assert probs[0].item() < 1e-10
    assert probs[2].item() < 1e-10
    assert abs((probs[1] + probs[3]).item() - 1.0) < 1e-5


def test_poe_novel_slot_gets_full_mass():
    """When all known hypotheses are eliminated, H_novel takes 100% of the probability."""
    enc = _tiny_encoder()
    model = RexDecisionModel(enc, tools=["tool_a"], head_layers=1, n_act=2)
    model.eval()

    input_ids, attention_mask, marker_pos, marker_mask, _ = _dummy_inputs(batch=1, n_markers=4)
    # H0, H1, H2 eliminated; only slot 3 (H_novel) alive
    hyp_alive = torch.tensor([[False, False, False, True]], dtype=torch.bool)

    with torch.no_grad():
        out = model.rex_forward(input_ids, attention_mask, marker_pos, marker_mask, hyp_alive=hyp_alive)
        probs = torch.softmax(out["verdict_logits"][0], dim=-1)

    assert abs(probs[3].item() - 1.0) < 1e-5
    assert not torch.isnan(probs).any()


def test_poe_all_eliminated_raises_error():
    """If all hypotheses including H_novel are marked dead, raise ValueError instead of returning NaN."""
    enc = _tiny_encoder()
    model = RexDecisionModel(enc, tools=["tool_a"], head_layers=1, n_act=2)

    input_ids, attention_mask, marker_pos, marker_mask, _ = _dummy_inputs(batch=1, n_markers=3)
    hyp_alive = torch.tensor([[False, False, False]], dtype=torch.bool)

    failed = False
    try:
        model.rex_forward(input_ids, attention_mask, marker_pos, marker_mask, hyp_alive=hyp_alive)
    except ValueError as e:
        failed = True
        assert "All hypotheses have been eliminated" in str(e)
    assert failed, "Expected ValueError when all hypotheses are eliminated"


def test_tool_mask_disables_tools():
    """Disabled tools receive -1e4 in tool_logits."""
    enc = _tiny_encoder()
    model = RexDecisionModel(enc, tools=["t0", "t1", "t2"], head_layers=1, n_act=2)
    model.eval()

    input_ids, attention_mask, marker_pos, marker_mask, _ = _dummy_inputs(batch=1, n_markers=2)
    tool_mask = torch.tensor([[True, False, True]], dtype=torch.bool)

    with torch.no_grad():
        out = model.rex_forward(input_ids, attention_mask, marker_pos, marker_mask, tool_mask=tool_mask)
        t_logits = out["tool_logits"][0]

    assert t_logits[1].item() == -1e4
    assert t_logits[0].item() > -1e3
    assert t_logits[2].item() > -1e3


def test_rex_decision_model_tool_validation():
    """RexDecisionModel validates that tools list is not empty, has no duplicates or empty strings."""
    enc = _tiny_encoder()

    # Empty tools
    failed = False
    try:
        RexDecisionModel(enc, tools=[])
    except ValueError as e:
        failed = True
        assert "tools list must not be empty" in str(e)
    assert failed

    # Duplicate tools
    failed = False
    try:
        RexDecisionModel(enc, tools=["tool_a", "tool_b", "tool_a"])
    except ValueError as e:
        failed = True
        assert "Duplicate tool names found" in str(e)
    assert failed

    # Empty string tool
    failed = False
    try:
        RexDecisionModel(enc, tools=["tool_a", "  "])
    except ValueError as e:
        failed = True
        assert "Tool names must be non-empty strings" in str(e)
    assert failed


# ==============================================================================
# 4. Vocabulary Expansion & Warm-Start Tests (Step 2.1)
# ==============================================================================

def test_add_type_tokens_warm_start():
    """New type tokens are warm-started using mean embedding of their descriptive words."""
    tok = _get_offline_tok()
    enc = _tiny_encoder(vocab_size=len(tok))
    type_tokens = {
        "<TYPE_PROMO_ABUSE>": "promotion abuse burst",
        "<TYPE_LATENCY>": "latency error",
    }

    token_map = add_type_tokens(tok, enc, type_tokens)

    assert "<TYPE_PROMO_ABUSE>" in token_map
    assert "<TYPE_LATENCY>" in token_map

    promo_id = token_map["<TYPE_PROMO_ABUSE>"]
    check_type_token(tok, "<TYPE_PROMO_ABUSE>")

    embeds = enc.get_input_embeddings().weight
    words = tok("promotion abuse burst", add_special_tokens=False)["input_ids"]
    expected_mean = embeds[words].mean(dim=0)

    assert torch.allclose(embeds[promo_id], expected_mean, atol=1e-5)


def test_check_type_token_validation():
    """check_type_token validates presence and single-token integrity."""
    tok = _get_offline_tok()
    tok.add_tokens(["<TYPE_VALID>"], special_tokens=True)
    tid = check_type_token(tok, "<TYPE_VALID>")
    assert tid >= 0

    failed = False
    try:
        check_type_token(tok, "")
    except ValueError as e:
        failed = True
        assert "Type token must be a non-empty string" in str(e)
    assert failed


# ==============================================================================
# 5. Input Layout Formatter & Collation Tests (Step 2.2)
# ==============================================================================

def test_encode_rex_input_layout():
    """encode_rex_input formats template and extracts hypothesis markers."""
    tok = _get_offline_tok()
    tok.add_tokens(["<TYPE_PROMO>"], special_tokens=True)
    hypotheses = {
        "H_botnet": "automated botnet traffic",
        "H_sybil": "distributed device farm",
        "H_shared": "residential proxy pool",
        "H_novel": "unexplained root cause",
    }

    item = encode_rex_input(
        tok,
        type_token="<TYPE_PROMO>",
        instruction="which hypothesis explains this anomaly?",
        hypotheses=hypotheses,
        state="observation error spike",
        alive={"H_botnet": False},
    )

    assert len(item["markers"]) == 4
    assert item["hyp_keys"] == ["H_botnet", "H_sybil", "H_shared", "H_novel"]
    assert item["hyp_alive"] == [False, True, True, True]
    assert item["input_ids"][0] == tok.cls_token_id


def test_encode_rex_input_guards():
    """encode_rex_input strictly enforces H_novel presence and non-elimination."""
    tok = _get_offline_tok()
    tok.add_tokens(["<TYPE_PROMO>"], special_tokens=True)

    # Missing H_novel at end
    failed1 = False
    try:
        encode_rex_input(
            tok,
            "<TYPE_PROMO>",
            "instruction",
            hypotheses={"H_0": "desc 0", "H_1": "desc 1"},
        )
    except ValueError as e:
        failed1 = True
        assert "must be 'H_novel'" in str(e)
    assert failed1

    # Attempting to eliminate H_novel
    failed2 = False
    try:
        encode_rex_input(
            tok,
            "<TYPE_PROMO>",
            "instruction",
            hypotheses={"H_0": "desc 0", "H_novel": "residual"},
            alive={"H_novel": False},
        )
    except ValueError as e:
        failed2 = True
        assert "Slot 'H_novel' cannot be eliminated" in str(e)
    assert failed2

    # Typo / unknown key in alive mask
    failed3 = False
    try:
        encode_rex_input(
            tok,
            "<TYPE_PROMO>",
            "instruction",
            hypotheses={"H_0": "desc 0", "H_novel": "residual"},
            alive={"H_typo": False},
        )
    except ValueError as e:
        failed3 = True
        assert "Unknown hypothesis keys in alive mask" in str(e)
    assert failed3


def test_collate_rex():
    """collate_rex correctly pads variable length sequences and marker positions."""
    tok = _get_offline_tok()
    tok.add_tokens(["<TYPE_PROMO>"], special_tokens=True)
    hypotheses = {"H_0": "desc 0", "H_novel": "residual"}

    item1 = encode_rex_input(tok, "<TYPE_PROMO>", "short query", hypotheses, state="s1")
    item2 = encode_rex_input(tok, "<TYPE_PROMO>", "longer query with more text", hypotheses, state="s2 s3 s4")

    batch = collate_rex([item1, item2], pad_token_id=tok.pad_token_id)

    assert batch["input_ids"].shape[0] == 2
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["marker_pos"].shape == (2, 2)
    assert batch["hyp_alive"].shape == (2, 2)


# ==============================================================================
# 6. Checkpoint Conversion & Config Tests (Step 2.5)
# ==============================================================================

def test_from_laya_conversion():
    """from_rex transfers weights from DecisionModel to RexDecisionModel cleanly."""
    enc = _tiny_encoder()
    laya = DecisionModel(enc, head_layers=1, n_act=2)
    tools = ["tool_1", "tool_2", "tool_3"]

    rex = from_rex(laya, tools=tools, tool_hidden=64, tool_prior=0.3)

    assert isinstance(rex, RexDecisionModel)
    assert rex.tools == tools
    assert rex.tool_embeddings.shape[0] == len(tools)

    # Check that scorer weights match exactly
    assert torch.allclose(rex.scorer[1].weight, laya.scorer[1].weight)


def test_build_model_rex_arch():
    """build_model constructs RexDecisionModel when arch=='rex' or 'rex2'."""
    cfg = {
        "arch": "rex2",
        "encoder": "bert-base-uncased",
        "head_layers": 1,
        "rex": {
            "tools": ["t1", "t2"],
            "type_tokens": {"<TYPE_A>": "description a"},
            "tool_hidden": 32,
            "tool_dropout": 0.05,
            "tool_prior": 0.1,
        },
    }
    enc = _tiny_encoder()
    model = RexDecisionModel(enc, tools=cfg["rex"]["tools"], head_layers=1)
    conf = rex_config(model)

    assert conf["arch"] in ("rex", "rex2")
    assert conf["rex"]["tools"] == ["t1", "t2"]


def test_rex_checkpoint_round_trip():
    """Verify that a converted Rex checkpoint on disk can be loaded back via load_checkpoint and Agent."""
    import shutil
    from rex.agent import Agent
    from rex.train import load_checkpoint, save_checkpoint

    tok = _get_offline_tok()
    laya_dir = tempfile.mkdtemp(prefix="test_laya_chkpt_")
    rex_out = tempfile.mkdtemp(prefix="test_rex_out_")

    try:
        # 1. Create a dummy trained Laya checkpoint on disk with distinct known weights
        enc = _tiny_encoder(vocab_size=len(tok))
        laya_model = DecisionModel(enc, head_layers=1, n_act=2)
        with torch.no_grad():
            nn.init.constant_(laya_model.scorer[1].weight, 3.14159)
            nn.init.constant_(laya_model.act_head[0].weight, 2.71828)

        base_cfg = {
            "encoder": "bert",
            "head_layers": 1,
            "act_costs": {"escalate": 1.0},
        }
        # Save Laya checkpoint using standard save_checkpoint
        save_checkpoint(laya_model, tok, base_cfg, laya_dir)

        # 2. Load Laya checkpoint using load_checkpoint
        loaded_laya, loaded_tok, loaded_cfg = load_checkpoint(laya_dir)
        assert torch.allclose(loaded_laya.scorer[1].weight, torch.tensor(3.14159), atol=1e-3)
        assert torch.allclose(loaded_laya.act_head[0].weight, torch.tensor(2.71828), atol=1e-3)

        # 3. Add type tokens (expanding vocab and embeddings) and convert to Rex
        type_tokens = {"<TYPE_SPECIAL>": "special event burst"}
        add_type_tokens(loaded_tok, loaded_laya, type_tokens)

        tools = ["ip_reputation", "auth_history"]
        rex_model = from_rex(loaded_laya, tools=tools, type_tokens=type_tokens)

        # Save Rex checkpoint using save_rex_checkpoint (which writes encoder/ with updated vocab)
        save_rex_checkpoint(rex_model, loaded_tok, rex_out, base_cfg=loaded_cfg)

        # 4. Verify reload via load_checkpoint
        reloaded_rex, reloaded_tok, reloaded_cfg = load_checkpoint(rex_out)
        assert isinstance(reloaded_rex, RexDecisionModel)
        assert reloaded_cfg["arch"] in ("rex", "rex2")
        assert reloaded_rex.tools == tools
        assert torch.allclose(reloaded_rex.scorer[1].weight, torch.tensor(3.14159), atol=1e-3)
        assert torch.allclose(reloaded_rex.act_head[0].weight, torch.tensor(2.71828), atol=1e-3)

        # 5. Verify reload via Agent
        agent = Agent(rex_out)
        assert isinstance(agent.model, RexDecisionModel)
    finally:
        shutil.rmtree(laya_dir, ignore_errors=True)
        shutil.rmtree(rex_out, ignore_errors=True)


if __name__ == "__main__":
    tests = [
        test_contextualize_and_decide_equivalence,
        test_contextualize_detach_encoder,
        test_tool_head_shapes_and_dtype,
        test_tool_head_prior_bias_init,
        test_tool_head_tool_embeddings_property,
        test_rex_decision_model_forward,
        test_rex_forward_outputs,
        test_poe_elimination_hard_masking,
        test_poe_novel_slot_gets_full_mass,
        test_poe_all_eliminated_raises_error,
        test_tool_mask_disables_tools,
        test_rex_decision_model_tool_validation,
        test_add_type_tokens_warm_start,
        test_check_type_token_validation,
        test_encode_rex_input_layout,
        test_encode_rex_input_guards,
        test_collate_rex,
        test_from_laya_conversion,
        test_build_model_rex_arch,
        test_rex_checkpoint_round_trip,
    ]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("\nAll Rex model architecture & tokenizer tests passed!")
