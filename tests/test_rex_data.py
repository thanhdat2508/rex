"""Comprehensive offline unit tests for Rex Data Pipeline & Investigation Engine (Part 3).

Tests follow F.I.R.S.T. principles and verify:
1. Registry loading and strict schema validations (missing H_novel, unknown op, orphan metric).
2. Popperian falsification evaluation and proof traces.
3. Tool masking under hypothesis elimination.
4. Trajectory rollouts and funnel budget scheduling (Step 3.1 & 3.2).
5. OOD / H_novel generation and full PoE elimination (Step 3.3).
6. Custom Dataset & Collate Function with full batch tensor contracts (Step 3.4).
7. End-to-end autograd training step with RexDecisionModel and RexMultiTaskLoss.
"""
from contextlib import contextmanager
import os
import sys
import tempfile

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, BertTokenizer


@contextmanager
def assert_raises(exc_type, match=None):
    failed = False
    try:
        yield
    except exc_type as e:
        failed = True
        if match:
            assert match in str(e), f"Expected '{match}' in error message '{str(e)}'"
    assert failed, f"Expected exception {exc_type.__name__} to be raised"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rex.losses import RexMultiTaskInputs, RexMultiTaskLoss
from rex.rex_data import (
    RexTrajectoryDataset,
    collate_rex_train,
    format_state,
    generate,
    oracle_tools,
    read_jsonl,
    rollout,
    sample_world,
    split_by_case,
    validate_sample,
    write_jsonl,
)
from rex.rex_model import RexDecisionModel
from rex.rex_registry import (
    FalsificationRule,
    load_registry,
)


def _get_offline_tok():
    """Create a fast offline BertTokenizer with required test tokens."""
    fd, vocab_path = tempfile.mkstemp(suffix=".txt", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(
            "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n"
            "<TYPE_PROMO_ABUSE>\n<TYPE_ACCOUNT_TAKEOVER>\n<TYPE_CREDENTIAL_STUFFING>\n"
            "<TYPE_API_SCRAPING>\n<TYPE_PAYMENT_FRAUD>\n"
            "which\nhypothesis\nexplains\nthe\nsignup\nburst\n"
            "promotion\nabuse\nbotnet\nsybil\nshared\nnovel\nattack\nfarm\n"
            "alert\nround_1\nround_2\nround_3\nstate\nevent\nlatency\nerror\n"
            "ip_risk_score\ndatacenter_ip_ratio\nkeystroke_entropy\ncarrier_nat\n"
        )
    tok = BertTokenizer(vocab_path)
    os.remove(vocab_path)
    tok.add_tokens(
        [
            "<TYPE_PROMO_ABUSE>",
            "<TYPE_ACCOUNT_TAKEOVER>",
            "<TYPE_CREDENTIAL_STUFFING>",
            "<TYPE_API_SCRAPING>",
            "<TYPE_PAYMENT_FRAUD>",
        ],
        special_tokens=True,
    )
    return tok


def _tiny_encoder(hidden_size: int = 32, vocab_size: int = 50) -> nn.Module:
    """Create a minimal 1-layer BERT encoder for rapid offline integration tests."""
    cfg = AutoConfig.for_model(
        "bert",
        hidden_size=hidden_size,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        vocab_size=vocab_size,
    )
    return AutoModel.from_config(cfg)


def _minimal_valid_registry_dict():
    return {
        "tools": ["tool_macro", "tool_micro"],
        "type_tokens": {"<TYPE_PROMO_ABUSE>": "promo abuse domain"},
        "metrics": {
            "alert_cnt": {"kind": "int", "min": 0, "max": 100},
            "metric_macro": {"kind": "float", "min": 0.0, "max": 1.0},
            "metric_micro": {"kind": "float", "min": 0.0, "max": 1.0},
        },
        "tool_specs": {
            "tool_macro": {"returns": ["metric_macro"], "scope": "macro"},
            "tool_micro": {"returns": ["metric_micro"], "scope": "micro"},
        },
        "scope_schedule": ["macro", "micro"],
        "domains": {
            "<TYPE_PROMO_ABUSE>": {
                "instruction": "which hypothesis explains the signup burst",
                "alert_metrics": ["alert_cnt"],
                "hypotheses": [
                    {
                        "id": "H_known_1",
                        "text": "botnet attack",
                        "associated_tools": ["tool_macro"],
                        "falsification_rules": [
                            {"metric": "metric_macro", "operator": "GT", "threshold": 0.5}
                        ],
                    },
                    {
                        "id": "H_known_2",
                        "text": "sybil farm",
                        "associated_tools": ["tool_micro"],
                        "falsification_rules": [
                            {"metric": "metric_micro", "operator": "LT", "threshold": 0.3}
                        ],
                    },
                    {
                        "id": "H_novel",
                        "text": "novel event",
                        "is_residual": True,
                    },
                ],
            }
        },
    }


# ==============================================================================
# 1. Registry & Schema Validation Tests
# ==============================================================================

def test_load_registry_valid():
    """Valid registry dictionary loads properly into structured Registry."""
    reg = load_registry(_minimal_valid_registry_dict())
    assert len(reg.tools) == 2
    assert "<TYPE_PROMO_ABUSE>" in reg.domains
    assert len(reg.metrics) == 3


def test_load_registry_missing_h_novel_raises():
    """Registry without H_novel must raise ValueError."""
    data = _minimal_valid_registry_dict()
    data["domains"]["<TYPE_PROMO_ABUSE>"]["hypotheses"].pop()  # remove H_novel
    with assert_raises(ValueError, match="H_novel"):
        load_registry(data)


def test_load_registry_h_novel_not_last_raises():
    """H_novel not positioned at the end of hypotheses list must raise ValueError."""
    data = _minimal_valid_registry_dict()
    hyps = data["domains"]["<TYPE_PROMO_ABUSE>"]["hypotheses"]
    hyps[0], hyps[2] = hyps[2], hyps[0]  # swap H_novel to index 0
    with assert_raises(ValueError, match="final index"):
        load_registry(data)


def test_load_registry_known_hypothesis_no_rules_raises():
    """A known hypothesis without any falsification rule cannot be eliminated and must raise ValueError."""
    data = _minimal_valid_registry_dict()
    data["domains"]["<TYPE_PROMO_ABUSE>"]["hypotheses"][0]["falsification_rules"] = []
    with assert_raises(ValueError, match="no falsification rules"):
        load_registry(data)


def test_load_registry_unknown_operator_raises():
    """Unknown falsification operator must raise ValueError."""
    data = _minimal_valid_registry_dict()
    data["domains"]["<TYPE_PROMO_ABUSE>"]["hypotheses"][0]["falsification_rules"][0]["operator"] = "UNKNOWN_OP"
    with assert_raises(ValueError, match="Unknown falsification operator"):
        load_registry(data)


def test_load_registry_orphan_metric_raises():
    """Rule on a metric neither in alert nor returned by associated tools must raise ValueError."""
    data = _minimal_valid_registry_dict()
    # tool_macro only returns metric_macro, but rule references metric_micro
    data["domains"]["<TYPE_PROMO_ABUSE>"]["hypotheses"][0]["falsification_rules"][0]["metric"] = "metric_micro"
    with assert_raises(ValueError, match="neither in domain alert_metrics nor returned"):
        load_registry(data)


def test_load_registry_unknown_tool_raises():
    """Hypothesis referencing an unlisted tool must raise ValueError."""
    data = _minimal_valid_registry_dict()
    data["domains"]["<TYPE_PROMO_ABUSE>"]["hypotheses"][0]["associated_tools"] = ["non_existent_tool"]
    with assert_raises(ValueError, match="references unknown tool"):
        load_registry(data)


# ==============================================================================
# 2. Popperian Falsification & Elimination Tests
# ==============================================================================

def test_falsify_operators_evaluation():
    """Test all logical comparison operators for Popperian falsification."""
    rule_gt = FalsificationRule("m", "GT", 0.5)
    assert rule_gt.eval(0.6) is True
    assert rule_gt.eval(0.4) is False

    rule_lt = FalsificationRule("m", "LT", 0.5)
    assert rule_lt.eval(0.4) is True
    assert rule_lt.eval(0.6) is False

    rule_gte = FalsificationRule("m", "GTE", 0.5)
    assert rule_gte.eval(0.5) is True
    assert rule_gte.eval(0.49) is False

    rule_lte = FalsificationRule("m", "LTE", 0.5)
    assert rule_lte.eval(0.5) is True
    assert rule_lte.eval(0.51) is False

    rule_eq = FalsificationRule("m", "EQ", 0)
    assert rule_eq.eval(0) is True
    assert rule_eq.eval(1) is False

    rule_neq = FalsificationRule("m", "NEQ", 0)
    assert rule_neq.eval(1) is True
    assert rule_neq.eval(0) is False


def test_falsify_h_novel_never_eliminated():
    """H_novel must unconditionally remain alive across all turns."""
    reg = load_registry(_minimal_valid_registry_dict())
    obs = {"metric_macro": 0.9, "metric_micro": 0.1}
    alive, _ = reg.falsify("<TYPE_PROMO_ABUSE>", obs)
    assert alive["H_novel"] is True


def test_falsify_proof_trace_recorded():
    """When a hypothesis is eliminated, eliminated_by must record the triggering rule and value."""
    reg = load_registry(_minimal_valid_registry_dict())
    obs = {"metric_macro": 0.8}  # Triggers H_known_1 (GT 0.5)
    alive, eliminated_by = reg.falsify("<TYPE_PROMO_ABUSE>", obs)
    assert alive["H_known_1"] is False
    assert "H_known_1" in eliminated_by
    assert "metric_macro GT 0.5 (observed 0.8)" in eliminated_by["H_known_1"][0]


def test_falsify_preserves_prior_eliminated():
    """A hypothesis eliminated in round t-1 must remain eliminated in round t even without new observations."""
    reg = load_registry(_minimal_valid_registry_dict())
    prior_alive = {"H_known_1": False, "H_known_2": True, "H_novel": True}
    alive, _ = reg.falsify("<TYPE_PROMO_ABUSE>", observations={}, current_alive=prior_alive)
    assert alive["H_known_1"] is False


# ==============================================================================
# 3. Tool Masking Tests
# ==============================================================================

def test_tool_mask_disables_tools_for_dead_hypotheses():
    """When H_known_1 is dead, tool_macro (associated only with H_known_1) must be masked out."""
    reg = load_registry(_minimal_valid_registry_dict())
    alive = {"H_known_1": False, "H_known_2": True, "H_novel": True}
    mask = reg.tool_mask("<TYPE_PROMO_ABUSE>", alive)
    # tools: [tool_macro, tool_micro]
    assert mask[0] is False  # tool_macro disabled
    assert mask[1] is True   # tool_micro active


def test_tool_mask_disables_already_called_tools():
    """A tool already dispatched in a previous round must be masked out to prevent redundant invocation."""
    reg = load_registry(_minimal_valid_registry_dict())
    alive = {"H_known_1": True, "H_known_2": True, "H_novel": True}
    mask = reg.tool_mask("<TYPE_PROMO_ABUSE>", alive, called_tools=["tool_macro"])
    assert mask[0] is False  # tool_macro already called
    assert mask[1] is True


# ==============================================================================
# 4. Trajectory Generation & Funnel Budget Tests (Steps 3.1 & 3.2)
# ==============================================================================

def test_sample_world_in_distribution_preserves_target():
    """In an in-distribution sample, the target H* must never have its rules falsified."""
    reg = load_registry(_minimal_valid_registry_dict())
    import random
    rng = random.Random(42)
    world = sample_world(reg, "<TYPE_PROMO_ABUSE>", rng, novel_ratio=0.0)
    target = world["target"]
    assert target != "H_novel"

    # Flatten all world observations
    all_obs = dict(world["alert"])
    for t_data in world["tool_data"].values():
        all_obs.update(t_data)

    alive, _ = reg.falsify("<TYPE_PROMO_ABUSE>", all_obs)
    assert alive[target] is True, "Target hypothesis was falsified in world generation!"


def test_sample_world_ood_novel_falsifies_all_known():
    """Step 3.3: In OOD / H_novel cases, all known hypotheses must be falsified by observations."""
    reg = load_registry(_minimal_valid_registry_dict())
    import random
    rng = random.Random(42)
    world = sample_world(reg, "<TYPE_PROMO_ABUSE>", rng, novel_ratio=1.0)
    assert world["target"] == "H_novel"

    all_obs = dict(world["alert"])
    for t_data in world["tool_data"].values():
        all_obs.update(t_data)

    alive, _ = reg.falsify("<TYPE_PROMO_ABUSE>", all_obs)
    assert alive["H_known_1"] is False
    assert alive["H_known_2"] is False
    assert alive["H_novel"] is True


def test_oracle_tools_respects_budgets_and_scope():
    """Step 3.2: Round 1 chooses macro tool; round 2 chooses micro tool."""
    reg = load_registry(_minimal_valid_registry_dict())
    state = {"alert": {"alert_cnt": 50}}
    alive = {"H_known_1": True, "H_known_2": True, "H_novel": True}

    # Round 1: Scope schedule specifies 'macro', budget 3 -> chooses tool_macro
    t1_tools = oracle_tools(reg, "<TYPE_PROMO_ABUSE>", state, alive, t=1, budgets=(3, 2, 1))
    assert t1_tools == ["tool_macro"]

    # Round 2: Scope schedule specifies 'micro', budget 2 -> chooses tool_micro
    state["round_1"] = {"tool_macro": {"metric_macro": 0.4}}
    t2_tools = oracle_tools(reg, "<TYPE_PROMO_ABUSE>", state, alive, t=2, budgets=(3, 2, 1))
    assert t2_tools == ["tool_micro"]


def test_oracle_tools_returns_empty_when_no_active_hyps():
    """When all hypotheses have been inspected or resolved, oracle returns [] (stop calling tools)."""
    reg = load_registry(_minimal_valid_registry_dict())
    state = {"alert": {"alert_cnt": 50}}
    # Only H_novel alive
    alive = {"H_known_1": False, "H_known_2": False, "H_novel": True}
    tools = oracle_tools(reg, "<TYPE_PROMO_ABUSE>", state, alive, t=1)
    assert tools == []


def test_rollout_produces_valid_trajectory():
    """Step 3.1: Incident rollout produces 1-3 step samples plus a terminal sample with t in {1, 2, 3}."""
    reg = load_registry(_minimal_valid_registry_dict())
    import random
    rng = random.Random(123)
    samples = rollout(reg, "<TYPE_PROMO_ABUSE>", rng, case_id="case-0001")

    assert len(samples) >= 2  # At least 1 round + terminal
    terminal = samples[-1]
    assert terminal["tool_labels"] == []  # Terminal step calls no tools
    assert terminal["label_scope"] == "terminal"

    # Invariant: Every sample must have t in {1, 2, 3}
    assert all(s["t"] in (1, 2, 3) for s in samples)
    # When oracle stops at round t, terminal sample carries t (not t + 1)
    if len(samples) == 2:
        # Round 1 called tools, round 2 oracle returned empty -> terminal sample t is 2
        assert terminal["t"] == 2
    elif len(samples) == 3:
        assert terminal["t"] == 3


def test_payment_fraud_has_macro_tools_in_round_1():
    """Step 3.2: PAYMENT_FRAUD domain associates non-alert macro tools (ip_reputation, network_trace, geo_ip_lookup)."""
    reg = load_registry("examples/rex/registry.example.json")
    pf = reg.get_domain("<TYPE_PAYMENT_FRAUD>")
    card_testing = pf.hypothesis_map["H_card_testing"]
    merchant = pf.hypothesis_map["H_compromised_merchant"]

    # H_card_testing uses macro tool ip_reputation with datacenter_ip_ratio rule
    assert "ip_reputation" in card_testing.associated_tools
    rule_metrics = [r.metric for r in card_testing.falsification_rules]
    assert "datacenter_ip_ratio" in rule_metrics
    # Not tied to redundant transaction_velocity
    assert "transaction_velocity" not in card_testing.associated_tools

    # H_compromised_merchant uses macro tools geo_ip_lookup and network_trace
    assert "geo_ip_lookup" in merchant.associated_tools
    assert "network_trace" in merchant.associated_tools

    # Oracle in round 1 selects macro tools with genuine information gain
    state = {"alert": {"declined_rate": 0.4, "velocity_1h": 200}}
    alive = {h.id: True for h in pf.hypotheses}
    r1_tools = oracle_tools(reg, "<TYPE_PAYMENT_FRAUD>", state, alive, t=1, budgets=(3, 2, 1))

    assert len(r1_tools) >= 2
    assert all(reg.tool_specs[t].scope == "macro" for t in r1_tools)
    assert "ip_reputation" in r1_tools
    assert "transaction_velocity" not in r1_tools


def test_validate_sample_catches_target_eliminated():
    """validate_sample must raise ValueError if ground truth target is marked eliminated."""
    reg = load_registry(_minimal_valid_registry_dict())
    invalid_sample = {
        "case_id": "c1",
        "t": 1,
        "type_token": "<TYPE_PROMO_ABUSE>",
        "state": {},
        "alive": {"H_known_1": False, "H_novel": True},
        "allowed_tools": ["tool_macro"],
        "tool_labels": ["tool_macro"],
        "target": "H_known_1",
    }
    with assert_raises(ValueError, match="was marked eliminated"):
        validate_sample(invalid_sample, reg)


def test_validate_sample_catches_disabled_tool_selected():
    """validate_sample must raise ValueError if a disabled tool is labeled."""
    reg = load_registry(_minimal_valid_registry_dict())
    invalid_sample = {
        "case_id": "c1",
        "t": 1,
        "type_token": "<TYPE_PROMO_ABUSE>",
        "state": {},
        "alive": {"H_known_1": True, "H_novel": True},
        "allowed_tools": ["tool_macro"],
        "tool_labels": ["tool_micro"],  # Not in allowed_tools
        "target": "H_known_1",
    }
    with assert_raises(ValueError, match="not in allowed_tools"):
        validate_sample(invalid_sample, reg)


def test_validate_sample_catches_budget_overflow():
    """validate_sample must raise ValueError if tool labels exceed round budget."""
    reg = load_registry(_minimal_valid_registry_dict())
    invalid_sample = {
        "case_id": "c1",
        "t": 3,
        "type_token": "<TYPE_PROMO_ABUSE>",
        "state": {},
        "alive": {"H_known_1": True, "H_novel": True},
        "allowed_tools": ["tool_macro", "tool_micro"],
        "tool_labels": ["tool_macro", "tool_micro"],  # 2 tools in round 3 (budget is 1)
        "target": "H_known_1",
    }
    with assert_raises(ValueError, match="exceeds round budget"):
        validate_sample(invalid_sample, reg, budgets=(3, 2, 1))


# ==============================================================================
# 5. Dataset Split & Utilities Tests
# ==============================================================================

def test_split_by_case_strict_isolation():
    """Train and val splits must have strictly zero overlap in case_id."""
    reg = load_registry(_minimal_valid_registry_dict())
    samples = generate(reg, n_cases=30, rng_or_seed=42)
    train_s, val_s = split_by_case(samples, val_ratio=0.2, seed=42)

    train_cases = {s["case_id"] for s in train_s}
    val_cases = {s["case_id"] for s in val_s}

    assert train_cases.isdisjoint(val_cases)
    assert len(train_cases) + len(val_cases) == 30


def test_format_state_structure():
    """format_state produces structured and ordered string representation."""
    state = {
        "alert": {"count": 100, "rate": 0.5},
        "round_1": {"ip_rep": {"risk": 0.8}},
        "round_2": {"jitter": {"entropy": 0.9}},
    }
    text = format_state(state)
    assert "alert: count=100 rate=0.5" in text
    assert "round_1: ip_rep {risk=0.8}" in text
    assert "round_2: jitter {entropy=0.9}" in text


def test_dataset_jsonl_round_trip():
    """write_jsonl and read_jsonl preserve sample dictionary integrity."""
    samples = [{"case_id": "case-01", "t": 1, "target": "H_novel", "alive": {"H_novel": True}}]
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        write_jsonl(path, samples)
        loaded = read_jsonl(path)
        assert loaded == samples
    finally:
        os.remove(path)


# ==============================================================================
# 6. Custom Dataset & Collate Function Tests (Step 3.4)
# ==============================================================================

def test_rex_trajectory_dataset_and_collate():
    """Step 3.4: DataLoader collate returns input_ids, attention_mask, mask_token_indices,

    tool_labels (multi-hot), poe_mask (binary), and target_verdict (scalar index).
    """
    reg = load_registry(_minimal_valid_registry_dict())
    tok = _get_offline_tok()
    samples = generate(reg, n_cases=5, rng_or_seed=42)

    ds = RexTrajectoryDataset(
        samples=samples,
        tok=tok,
        tools=reg.tools,
        registry=reg,
        max_len=128,
        head_max_len=64,
        shuffle_hypotheses=False,
    )
    assert len(ds) == len(samples)

    batch_items = [ds[0], ds[1]]
    batch = collate_rex_train(batch_items, tok.pad_token_id)

    # 1. input_ids & attention_mask [B, S]
    assert batch["input_ids"].shape == (2, batch["input_ids"].size(1))
    assert batch["attention_mask"].shape == batch["input_ids"].shape

    # 2. marker_pos and mask_token_indices alias [B, K]
    assert "marker_pos" in batch
    assert "mask_token_indices" in batch
    assert torch.equal(batch["marker_pos"], batch["mask_token_indices"])

    # 3. tool_labels multi-hot [B, N_tools] float32
    assert batch["tool_labels"].shape == (2, len(reg.tools))
    assert batch["tool_labels"].dtype == torch.float32

    # 4. poe_mask binary [B, K]
    assert "poe_mask" in batch
    assert batch["poe_mask"].shape == batch["marker_pos"].shape

    # 5. target_verdict scalar index [B] long
    assert batch["target_verdict"].shape == (2,)
    assert batch["target_verdict"].dtype == torch.long

    # 6. is_terminal boolean flag [B] bool
    assert "is_terminal" in batch
    assert batch["is_terminal"].dtype == torch.bool
    assert batch["is_terminal"].shape == (2,)


def test_rex_trajectory_dataset_shuffling_updates_target_verdict():
    """When hypotheses are permuted across epochs, target_verdict index updates accordingly."""
    reg = load_registry(_minimal_valid_registry_dict())
    tok = _get_offline_tok()
    samples = [
        {
            "case_id": "c1",
            "t": 1,
            "type_token": "<TYPE_PROMO_ABUSE>",
            "state": {"alert": {"alert_cnt": 10}},
            "alive": {"H_known_1": True, "H_known_2": True, "H_novel": True},
            "allowed_tools": ["tool_macro"],
            "tool_labels": ["tool_macro"],
            "target": "H_known_2",
        }
    ]

    ds = RexTrajectoryDataset(
        samples=samples,
        tok=tok,
        tools=reg.tools,
        registry=reg,
        max_len=128,
        shuffle_hypotheses=True,
    )

    indices = set()
    for ep in range(10):
        ds.set_epoch(ep)
        item = ds[0]
        indices.add(item["target_verdict"])

    # Over 10 epochs, H_known_2 should land in more than 1 distinct slot
    assert len(indices) > 1


# ==============================================================================
# 7. End-to-End Training Integration Test
# ==============================================================================

def test_end_to_end_training_step():
    """Verify that batches from collate_rex_train seamlessly plug into RexDecisionModel

    and RexMultiTaskLoss, allowing backpropagation and optimizer parameter update.
    """
    reg = load_registry(_minimal_valid_registry_dict())
    tok = _get_offline_tok()
    samples = generate(reg, n_cases=6, rng_or_seed=42)

    ds = RexTrajectoryDataset(
        samples=samples,
        tok=tok,
        tools=reg.tools,
        registry=reg,
        max_len=128,
        head_max_len=64,
    )
    batch = collate_rex_train([ds[0], ds[1]], tok.pad_token_id)

    # Initialize tiny model
    encoder = _tiny_encoder(hidden_size=16, vocab_size=len(tok))
    model = RexDecisionModel(
        encoder=encoder,
        tools=reg.tools,
        type_tokens=reg.type_tokens,
        tool_hidden=16,
        head_layers=1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = RexMultiTaskLoss()

    # Forward pass
    out = model.rex_forward(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        marker_pos=batch["marker_pos"],
        marker_mask=batch["marker_mask"],
        qtype=batch["qtype"],
        hyp_alive=batch["hyp_alive"],
        tool_mask=batch.get("tool_mask"),
    )

    # Multi-task loss computation
    loss_inputs = RexMultiTaskInputs(
        tool_logits=out["tool_logits"],
        tool_labels=batch["tool_labels"],
        verdict_logits=out["verdict_logits"],
        poe_mask=batch["poe_mask"],
        target_verdict=batch["target_verdict"],
        tool_projection_weight=model.tool_head.tool_embeddings,
    )
    breakdown = loss_fn(loss_inputs)

    assert breakdown.total_loss.item() > 0.0
    assert torch.isfinite(breakdown.total_loss)

    # Backward and optimizer step
    optimizer.zero_grad()
    breakdown.total_loss.backward()
    optimizer.step()

    # Check finite gradients on model parameters
    assert model.tool_head.proj_out.weight.grad is not None
    assert torch.isfinite(model.tool_head.proj_out.weight.grad).all()


# ==============================================================================
# Script Execution Runner
# ==============================================================================

if __name__ == "__main__":
    tests = [
        test_load_registry_valid,
        test_load_registry_missing_h_novel_raises,
        test_load_registry_h_novel_not_last_raises,
        test_load_registry_known_hypothesis_no_rules_raises,
        test_load_registry_unknown_operator_raises,
        test_load_registry_orphan_metric_raises,
        test_load_registry_unknown_tool_raises,
        test_falsify_operators_evaluation,
        test_falsify_h_novel_never_eliminated,
        test_falsify_proof_trace_recorded,
        test_falsify_preserves_prior_eliminated,
        test_tool_mask_disables_tools_for_dead_hypotheses,
        test_tool_mask_disables_already_called_tools,
        test_sample_world_in_distribution_preserves_target,
        test_sample_world_ood_novel_falsifies_all_known,
        test_oracle_tools_respects_budgets_and_scope,
        test_oracle_tools_returns_empty_when_no_active_hyps,
        test_rollout_produces_valid_trajectory,
        test_payment_fraud_has_macro_tools_in_round_1,
        test_validate_sample_catches_target_eliminated,
        test_validate_sample_catches_disabled_tool_selected,
        test_validate_sample_catches_budget_overflow,
        test_split_by_case_strict_isolation,
        test_format_state_structure,
        test_dataset_jsonl_round_trip,
        test_rex_trajectory_dataset_and_collate,
        test_rex_trajectory_dataset_shuffling_updates_target_verdict,
        test_end_to_end_training_step,
    ]

    print(f"Running {len(tests)} Rex data pipeline unit tests offline...\n")
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")

    print("\nAll Rex data pipeline tests passed successfully!")
