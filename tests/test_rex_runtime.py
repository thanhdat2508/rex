"""Unit tests for Rex Runtime Decision Engine and Popperian Falsification (Part 5).

Tests verify:
1. ToolHead.select_tools: Floor threshold (0.35), Dynamic Margin (0.15), and Budget Caps (3, 2, 1).
2. HardMaskingManager: Exclusive vs shared tool masking based on hypothesis survival.
3. PoEFalsificationEngine: Relational operators (LT, GT, EQ, NEQ) and H_novel survival invariant.
4. RexDecisionModel.compute_verdict_probabilities: Hard logit masking and probability normalization.
5. RexDecisionModel.investigate: Full autonomous multi-round investigation lifecycle and early termination.
"""

import os
import sys
import tempfile
from typing import Any, Dict, List

import pytest
import torch
from transformers import AutoConfig, AutoModel, BertTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rex.rex_model import RexDecisionModel, ToolHead
from rex.rex_poe import (
    ComparisonOperator,
    FalsificationRule,
    HardMaskingManager,
    InvestigationResult,
    PoEFalsificationEngine,
    TerminationReason,
)
from rex.rex_tools import dispatch_three_branch
from rex.rex_vocab import add_type_tokens


# -----------------------------------------------------------------------------
# Fixtures and helpers
# -----------------------------------------------------------------------------

def _create_offline_tok() -> BertTokenizer:
    """Create a fast in-memory BertTokenizer with domain type tokens."""
    fd, vocab_path = tempfile.mkstemp(suffix=".txt", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(
            "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n"
            "which\nhypothesis\nexplains\nthe\nincident\n"
            "botnet\nsybil\nshared\nnovel\nstate\n"
        )
    tok = BertTokenizer(vocab_path)
    os.remove(vocab_path)
    add_type_tokens(tok, None, {"<TYPE_PROMO_ABUSE>": "promotional abuse"})
    return tok


def _create_test_rex_model(d: int = 32, n_tools: int = 4) -> RexDecisionModel:
    """Create a lightweight offline RexDecisionModel."""
    cfg = AutoConfig.for_model(
        "bert",
        hidden_size=d,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        vocab_size=100,
    )
    encoder = AutoModel.from_config(cfg)
    tools = [f"tool_{i}" for i in range(n_tools)]
    model = RexDecisionModel(
        encoder=encoder,
        tools=tools,
        type_tokens={"<TYPE_PROMO_ABUSE>": "promo abuse description"},
        head_layers=1,
        n_act=2,
    )
    model.eval()
    return model


# -----------------------------------------------------------------------------
# 1. Tests for dispatch_three_branch (Step 5.1: Gating & Budget)
# -----------------------------------------------------------------------------

def test_select_tools_floor_threshold():
    """If tool gain falls below cost hurdle and round > 1, no tools are selected."""
    tools = ["t0", "t1", "t2", "t3"]
    logits = torch.tensor([0.1, 0.2, 0.3, 0.4])
    gains = torch.tensor([0.05, 0.05, 0.05, 0.05])
    tool_costs = {"t0": 0.5, "t1": 0.5, "t2": 0.5, "t3": 0.5}
    tool_levels = {t: "HEAVY" for t in tools}

    action, selected, _ = dispatch_three_branch(
        tool_names=tools,
        tool_logits=logits,
        tool_gains=gains,
        s_prob=torch.tensor([0.1]),
        r_prob=torch.tensor([0.1]),
        round_idx=2,
        tool_costs=tool_costs,
        tool_levels=tool_levels,
    )
    assert action == "STALL"
    assert selected == []


def test_select_tools_relative_margin_and_budget_cap():
    """Runners-up must qualify and selection must be capped at K_t."""
    tools = ["t0", "t1", "t2", "t3", "t4"]
    logits = torch.tensor([2.0, 1.8, 1.5, 0.5, -1.0])
    gains = torch.tensor([0.9, 0.85, 0.8, 0.6, 0.2])
    tool_costs = {t: 0.1 for t in tools}

    # Round 1: Budget cap K_1 = 3 -> top 3 selected
    _, sel_r1, _ = dispatch_three_branch(
        tool_names=tools, tool_logits=logits, tool_gains=gains,
        s_prob=torch.tensor([0.2]), r_prob=torch.tensor([0.1]),
        round_idx=1, tool_costs=tool_costs, budget_caps=(3, 2, 1)
    )
    assert sel_r1 == ["t0", "t1", "t2"]

    # Round 2: Budget cap K_2 = 2 -> top 2 selected
    _, sel_r2, _ = dispatch_three_branch(
        tool_names=tools, tool_logits=logits, tool_gains=gains,
        s_prob=torch.tensor([0.2]), r_prob=torch.tensor([0.1]),
        round_idx=2, tool_costs=tool_costs, budget_caps=(3, 2, 1)
    )
    assert sel_r2 == ["t0", "t1"]

    # Round 3: Budget cap K_3 = 1 -> top 1 selected
    _, sel_r3, _ = dispatch_three_branch(
        tool_names=tools, tool_logits=logits, tool_gains=gains,
        s_prob=torch.tensor([0.2]), r_prob=torch.tensor([0.1]),
        round_idx=3, tool_costs=tool_costs, budget_caps=(3, 2, 1)
    )
    assert sel_r3 == ["t0"]


def test_select_tools_poe_tool_mask_suppression():
    """Tools disabled by tool_mask (False) must receive -1e4 and never be selected."""
    tools = ["t0", "t1", "t2"]
    logits = torch.tensor([5.0, 5.0, 5.0])
    gains = torch.tensor([0.8, 0.8, 0.8])
    tool_mask = torch.tensor([False, True, False])

    _, selected, _ = dispatch_three_branch(
        tool_names=tools, tool_logits=logits, tool_gains=gains,
        s_prob=torch.tensor([0.2]), r_prob=torch.tensor([0.1]),
        round_idx=1, tool_mask=tool_mask,
    )
    assert selected == ["t1"]


# -----------------------------------------------------------------------------
# 2. Tests for HardMaskingManager (Step 5.3: Tool Mask Derivation)
# -----------------------------------------------------------------------------

def test_hard_masking_manager_exclusive_and_shared_tools():
    """Verify tool masking rules: shared tools stay active if any hypothesis is alive."""
    manager = HardMaskingManager()
    tools = ["tool_botnet_exclusive", "tool_shared_bot_sybil", "tool_sybil_exclusive", "tool_unmapped"]
    tool_hyp_mapping = {
        "tool_botnet_exclusive": ["H_botnet"],
        "tool_shared_bot_sybil": ["H_botnet", "H_sybil"],
        "tool_sybil_exclusive": ["H_sybil"],
        # tool_unmapped has no entry
    }

    # Case 1: Both H_botnet and H_sybil are alive
    alive_1 = {"H_botnet": True, "H_sybil": True, "H_novel": True}
    mask_1 = manager.compute_tool_mask(tools, alive_1, tool_hyp_mapping)
    assert mask_1 == [True, True, True, True]

    # Case 2: H_botnet is eliminated, H_sybil is alive
    alive_2 = {"H_botnet": False, "H_sybil": True, "H_novel": True}
    mask_2 = manager.compute_tool_mask(tools, alive_2, tool_hyp_mapping)
    # tool_botnet_exclusive is disabled; tool_shared_bot_sybil remains active via H_sybil
    assert mask_2 == [False, True, True, True]

    # Case 3: Both H_botnet and H_sybil are eliminated
    alive_3 = {"H_botnet": False, "H_sybil": False, "H_novel": True}
    mask_3 = manager.compute_tool_mask(tools, alive_3, tool_hyp_mapping)
    # Both exclusive and shared tools are disabled; unmapped tool remains active
    assert mask_3 == [False, False, False, True]


# -----------------------------------------------------------------------------
# 3. Tests for PoEFalsificationEngine (Step 5.2: Karl Popper Falsification)
# -----------------------------------------------------------------------------

def test_poe_falsification_operators():
    """Test all comparison operators: LT, GT, EQ, NEQ."""
    engine = PoEFalsificationEngine()
    current_alive = {"H_bot": True, "H_sybil": True, "H_shared": True, "H_novel": True}

    rules = {
        "H_bot": [FalsificationRule(metric="keystroke_entropy", operator=ComparisonOperator.GT, threshold=0.8)],
        "H_sybil": [FalsificationRule(metric="device_count", operator=ComparisonOperator.LT, threshold=5.0)],
        "H_shared": [FalsificationRule(metric="cgnat_flag", operator=ComparisonOperator.EQ, threshold=0.0)],
    }

    # Telemetry:
    # keystroke_entropy = 0.95 (> 0.8 -> H_bot violated, eliminated)
    # device_count = 10.0 (not < 5.0 -> H_sybil consistent, alive)
    # cgnat_flag = 0.0 (== 0.0 -> H_shared violated, eliminated)
    obs = {"keystroke_entropy": 0.95, "device_count": 10.0, "cgnat_flag": 0.0}

    new_alive = engine.evaluate(obs, current_alive, rules)

    assert new_alive["H_bot"] is False
    assert new_alive["H_sybil"] is True
    assert new_alive["H_shared"] is False
    assert new_alive["H_novel"] is True


def test_poe_falsification_h_novel_immune():
    """H_novel must remain alive even if all other hypotheses are eliminated."""
    engine = PoEFalsificationEngine()
    current_alive = {"H_bot": True, "H_novel": True}
    rules = {
        "H_bot": [FalsificationRule(metric="cpu", operator=ComparisonOperator.GT, threshold=50.0)],
        "H_novel": [FalsificationRule(metric="cpu", operator=ComparisonOperator.GT, threshold=10.0)],
    }
    obs = {"cpu": 99.0}

    new_alive = engine.evaluate(obs, current_alive, rules)
    assert new_alive["H_bot"] is False
    # H_novel cannot be eliminated
    assert new_alive["H_novel"] is True


# -----------------------------------------------------------------------------
# 4. Tests for compute_verdict_probabilities (Step 5.3)
# -----------------------------------------------------------------------------

def test_compute_verdict_probabilities_normalizes_to_one():
    """Eliminated hypotheses must have 0.0 probability and survivors sum to 1.0."""
    model = _create_test_rex_model(d=16, n_tools=2)
    verdict_logits = torch.tensor([2.0, 1.0, 3.0, 0.5])
    hyp_keys = ["H_bot", "H_sybil", "H_shared", "H_novel"]
    hyp_alive = {"H_bot": True, "H_sybil": False, "H_shared": True, "H_novel": False}

    probs = model.compute_verdict_probabilities(verdict_logits, hyp_alive=hyp_alive, hyp_keys=hyp_keys)

    # Eliminated hypotheses must have 0.0 probability
    assert probs["H_sybil"] == 0.0
    assert probs["H_novel"] == 0.0

    # Surviving hypotheses must sum to 1.0
    active_sum = probs["H_bot"] + probs["H_shared"]
    assert pytest.approx(active_sum, rel=1e-5) == 1.0
    assert probs["H_shared"] > probs["H_bot"]  # logit 3.0 > 2.0


# -----------------------------------------------------------------------------
# 5. Tests for RexDecisionModel.investigate (Step 5.4: Full Autonomous Loop)
# -----------------------------------------------------------------------------

def test_investigate_early_termination_single_survivor():
    """Investigation should terminate in round 1 if only one hypothesis survives."""
    tok = _create_offline_tok()
    model = _create_test_rex_model(d=32, n_tools=3)
    model.tools = ["tool_0", "tool_1", "tool_2"]

    hypotheses = {
        "H_botnet": "automated scripted account creation",
        "H_sybil": "virtual device farm abuse",
        "H_novel": "unseen novel attack pattern",
    }

    # Rules designed so that round 1 eliminates H_botnet and H_novel is residual
    rules = {
        "H_botnet": [FalsificationRule(metric="entropy", operator=ComparisonOperator.GT, threshold=0.5)],
        "H_sybil": [FalsificationRule(metric="device_id", operator=ComparisonOperator.EQ, threshold=0.0)],
    }
    tool_hyp_mapping = {
        "tool_0": ["H_botnet"],
        "tool_1": ["H_sybil"],
        "tool_2": ["H_novel"],
    }

    # Dispatcher returns telemetry eliminating H_botnet; H_sybil survives
    def mock_dispatcher(tools: List[str]) -> Dict[str, Any]:
        return {"entropy": 0.9, "device_id": 1.0}

    result: InvestigationResult = model.investigate(
        tok=tok,
        type_token="<TYPE_PROMO_ABUSE>",
        instruction="Investigate massive signups",
        hypotheses=hypotheses,
        rules=rules,
        tool_hyp_mapping=tool_hyp_mapping,
        tool_dispatcher=mock_dispatcher,
        max_rounds=3,
    )

    assert isinstance(result, InvestigationResult)
    assert result.rounds_executed >= 1
    assert result.predicted_hypothesis in ("H_sybil", "H_novel")
    assert pytest.approx(sum(result.probabilities.values()), rel=1e-5) == 1.0


def test_investigate_all_cataloged_eliminated_triggers_escalation():
    """When all known hypotheses are falsified, system must escalate to human with H_novel."""
    tok = _create_offline_tok()
    model = _create_test_rex_model(d=32, n_tools=2)
    model.tools = ["tool_0", "tool_1"]

    hypotheses = {
        "H_botnet": "automated scripted account creation",
        "H_sybil": "virtual device farm abuse",
        "H_novel": "unseen novel attack pattern",
    }

    rules = {
        "H_botnet": [FalsificationRule(metric="anomaly", operator=ComparisonOperator.GT, threshold=0.5)],
        "H_sybil": [FalsificationRule(metric="anomaly", operator=ComparisonOperator.GT, threshold=0.5)],
    }
    tool_hyp_mapping = {
        "tool_0": ["H_botnet"],
        "tool_1": ["H_sybil"],
    }

    # Telemetry refutes both H_botnet and H_sybil
    def mock_dispatcher(tools: List[str]) -> Dict[str, Any]:
        return {"anomaly": 0.99}

    result = model.investigate(
        tok=tok,
        type_token="<TYPE_PROMO_ABUSE>",
        instruction="Zero day investigation",
        hypotheses=hypotheses,
        rules=rules,
        tool_hyp_mapping=tool_hyp_mapping,
        tool_dispatcher=mock_dispatcher,
        max_rounds=3,
    )

    assert result.is_escalated is True
    assert result.predicted_hypothesis == "H_novel"
    assert result.termination_reason == TerminationReason.ALL_CATALOGED_ELIMINATED
    assert result.probabilities["H_novel"] == 1.0


def test_investigate_empty_tool_selection_short_circuits():
    """If no tools clear the s_min confidence floor, loop short-circuits directly to verdict."""
    tok = _create_offline_tok()
    model = _create_test_rex_model(d=32, n_tools=2)

    hypotheses = {
        "H_botnet": "automated scripted account creation",
        "H_novel": "unseen novel attack pattern",
    }

    # Mock forward to simulate empty tool selection / VERDICT
    orig_forward = model.forward
    def mock_fwd(*args, **kwargs):
        v, s, out = orig_forward(*args, **kwargs)
        out.selected_tools = [[]]
        out.proposed_action = ["VERDICT"]
        return v, s, out
    model.forward = mock_fwd

    called = False

    def mock_dispatcher(tools: List[str]) -> Dict[str, Any]:
        nonlocal called
        called = True
        return {}

    result = model.investigate(
        tok=tok,
        type_token="<TYPE_PROMO_ABUSE>",
        instruction="Low signal prompt",
        hypotheses=hypotheses,
        rules={},
        tool_hyp_mapping={},
        tool_dispatcher=mock_dispatcher,
        max_rounds=3,
    )

    # Tool dispatcher must NOT be called when tool list is empty
    assert called is False
    assert result.termination_reason == TerminationReason.EMPTY_TOOL_SELECTION
