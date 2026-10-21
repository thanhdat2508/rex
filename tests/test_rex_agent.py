"""Unit tests for InvestigatorAgent and autonomous investigation loops (Step A10)."""
import os
import tempfile
from typing import Any, Dict, List
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, BertTokenizer

from rex.rex_agent import InvestigatorAgent
from rex.rex_model import RexDecisionModel
from rex.rex_poe import ComparisonOperator, FalsificationRule, TerminationReason
from rex.rex_vocab import add_type_tokens


def _create_offline_tok() -> BertTokenizer:
    """Create a fast in-memory BertTokenizer with domain type tokens."""
    fd, vocab_path = tempfile.mkstemp(suffix=".txt", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(
            "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n"
            "which\nhypothesis\nexplains\nthe\nincident\n"
            "bot\nsybil\nshared\nnovel\nstate\nanomaly\n"
        )
    tok = BertTokenizer(vocab_path)
    os.remove(vocab_path)
    add_type_tokens(tok, None, {"<TYPE_PROMO>": "promotional abuse", "<TYPE_AUTH>": "auth abuse"})
    return tok


def _create_mock_agent(n_tools: int = 3, d: int = 32) -> InvestigatorAgent:
    cfg = AutoConfig.for_model(
        "bert",
        hidden_size=d,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        vocab_size=200,
    )
    enc = AutoModel.from_config(cfg)
    tools = [f"tool_{i}" for i in range(n_tools)]
    model = RexDecisionModel(
        encoder=enc,
        tools=tools,
        type_tokens={"<TYPE_PROMO>": "promo", "<TYPE_AUTH>": "auth"},
        head_layers=1,
        n_act=2,
    )
    return InvestigatorAgent(model=model)


def test_investigator_agent_single_survivor_termination():
    """Verify that investigator agent terminates immediately when a single survivor remains."""
    agent = _create_mock_agent(n_tools=3)
    tok = _create_offline_tok()

    hypotheses = {
        "H_bot": "bot",
        "H_sybil": "sybil",
        "H_novel": "novel",
    }
    rules = {
        "H_bot": [FalsificationRule(metric="err_rate", operator=ComparisonOperator.GT, threshold=5.0)],
        "H_sybil": [],
    }
    tool_hyp_mapping = {
        "tool_0": ["H_bot"],
        "tool_1": ["H_sybil"],
        "tool_2": ["H_novel"],
    }

    # Dispatcher returns err_rate=10.0 (eliminates H_bot), leaving H_sybil as sole survivor
    def mock_dispatcher(tools: List[str]) -> Dict[str, Any]:
        return {"err_rate": 10.0}

    result = agent.investigate(
        tok=tok,
        type_token="<TYPE_PROMO>",
        instruction="incident",
        hypotheses=hypotheses,
        rules=rules,
        tool_hyp_mapping=tool_hyp_mapping,
        tool_dispatcher=mock_dispatcher,
        max_rounds=2,
    )

    assert result.rounds_executed == 1
    assert result.termination_reason == TerminationReason.SINGLE_SURVIVOR
    assert result.probabilities["H_bot"] == 0.0
    assert result.probabilities["H_sybil"] > 0.5
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-5


def test_investigator_agent_all_cataloged_eliminated_escalates():
    """Verify that when all known hypotheses are refuted, agent escalates to H_novel."""
    agent = _create_mock_agent(n_tools=2)
    tok = _create_offline_tok()

    hypotheses = {
        "H_bot": "bot",
        "H_sybil": "sybil",
        "H_novel": "novel",
    }
    rules = {
        "H_bot": [FalsificationRule(metric="anomaly", operator=ComparisonOperator.GT, threshold=0.5)],
        "H_sybil": [FalsificationRule(metric="anomaly", operator=ComparisonOperator.GT, threshold=0.5)],
    }
    tool_hyp_mapping = {
        "tool_0": ["H_bot"],
        "tool_1": ["H_sybil"],
    }

    def mock_dispatcher(tools: List[str]) -> Dict[str, Any]:
        return {"anomaly": 0.9}

    result = agent.investigate(
        tok=tok,
        type_token="<TYPE_AUTH>",
        instruction="incident",
        hypotheses=hypotheses,
        rules=rules,
        tool_hyp_mapping=tool_hyp_mapping,
        tool_dispatcher=mock_dispatcher,
        max_rounds=2,
    )

    assert result.termination_reason == TerminationReason.ALL_CATALOGED_ELIMINATED
    assert result.predicted_hypothesis == "H_novel"
    assert result.is_escalated is True
