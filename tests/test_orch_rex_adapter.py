"""Unit and parity tests for RexAdapter connecting RexDecisionModel to CaseRunner (Step B12)."""
import asyncio
import os
import sys
import tempfile
import time
from typing import Any, Dict, List

import torch
from transformers import AutoConfig, AutoModel, BertTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rex.orch.agent_port import Case
from rex.orch.guard import Guard
from rex.orch.rex_adapter import RexAdapter
from rex.orch.runner import CaseRunner
from rex.orch.state_store import InMemoryStateStore
from rex.orch.stop import StopConfig
from rex.orch.tool_client import ToolClient
from rex.orch.tool_runner import ToolRunner
from rex.orch.trace_store import InMemoryTraceStore
from rex.rex_agent import InvestigatorAgent
from rex.rex_model import RexDecisionModel
from rex.rex_poe import ComparisonOperator, FalsificationRule
from rex.rex_vocab import add_type_tokens
from tests.orch_fake_sources import FakeMCPSources


def _create_tok() -> BertTokenizer:
    fd, vocab_path = tempfile.mkstemp(suffix=".txt", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(
            "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n"
            "which\nhypothesis\nexplains\nthe\nincident\n"
            "bot\nsybil\nshared\nnovel\nstate\nanomaly\n"
            "ip\ndevice\ncountry\nfailed\n"
        )
    tok = BertTokenizer(vocab_path)
    os.remove(vocab_path)
    add_type_tokens(tok, None, {"<TYPE_PROMO>": "promo", "<TYPE_AUTH>": "auth"})
    return tok


def _create_model(d: int = 32) -> RexDecisionModel:
    torch.manual_seed(42)
    cfg = AutoConfig.for_model(
        "bert",
        hidden_size=d,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        vocab_size=200,
    )
    enc = AutoModel.from_config(cfg)
    tools = ["geo_ip_lookup", "device_fingerprint", "auth_history"]
    return RexDecisionModel(
        encoder=enc,
        tools=tools,
        type_tokens={"<TYPE_PROMO>": "promo", "<TYPE_AUTH>": "auth"},
        head_layers=1,
        n_act=2,
    )


def test_rex_adapter_parity_with_investigator_agent():
    """Verify RexAdapter produces identical turn-1 tool selections as InvestigatorAgent."""
    torch.manual_seed(100)
    tok = _create_tok()
    model = _create_model()

    hypotheses = {
        "H_bot": "bot activity",
        "H_sybil": "sybil attack",
        "H_novel": "novel attack",
    }
    rules = {
        "H_bot": [FalsificationRule(metric="country_mismatch", operator=ComparisonOperator.EQ, threshold=True)],
        "H_sybil": [],
    }
    tool_hyp_mapping = {
        "geo_ip_lookup": ["H_bot"],
        "device_fingerprint": ["H_sybil"],
        "auth_history": ["H_novel"],
    }

    # 1. Run InvestigatorAgent turn 1
    investigator = InvestigatorAgent(model=model)
    case_text = "Suspicious velocity across multiple regions"

    first_tools_investigator: List[str] = []

    def capturing_dispatcher(tools: List[str]) -> Dict[str, Any]:
        nonlocal first_tools_investigator
        first_tools_investigator = list(tools)
        return {"country_mismatch": True}

    investigator.investigate(
        tok=tok,
        type_token="<TYPE_PROMO>",
        instruction="Identify suspicious logins",
        hypotheses=hypotheses,
        rules=rules,
        tool_hyp_mapping=tool_hyp_mapping,
        tool_dispatcher=capturing_dispatcher,
        initial_state=case_text,
        max_rounds=1,
    )

    # 2. Run RexAdapter start
    adapter = RexAdapter(
        model=model,
        tok=tok,
        instruction="Identify suspicious logins",
        hypotheses=hypotheses,
        rules=rules,
        tool_hyp_mapping=tool_hyp_mapping,
    )

    case = Case(
        ticket_id="TCK-PARITY",
        type_token="<TYPE_PROMO>",
        text=case_text,
    )
    all_available = {t: True for t in model.tools}
    cmd = adapter.start(case, all_available)

    # Assert tool selection parity between InvestigatorAgent and RexAdapter
    assert cmd.tools == first_tools_investigator, f"Mismatch: {cmd.tools} != {first_tools_investigator}"


def test_rex_adapter_end_to_end_with_case_runner_and_latency():
    tok = _create_tok()
    model = _create_model()

    hypotheses = {
        "H_bot": "bot activity",
        "H_sybil": "sybil attack",
        "H_novel": "novel attack",
    }
    rules = {
        "H_bot": [FalsificationRule(metric="country_mismatch", operator=ComparisonOperator.EQ, threshold=True)],
    }
    tool_hyp_mapping = {
        "geo_ip_lookup": ["H_bot"],
        "device_fingerprint": ["H_sybil"],
        "auth_history": ["H_novel"],
    }

    adapter = RexAdapter(
        model=model,
        tok=tok,
        instruction="Identify suspicious logins",
        hypotheses=hypotheses,
        rules=rules,
        tool_hyp_mapping=tool_hyp_mapping,
    )

    mock_source = FakeMCPSources(artificial_latency_s=0.002)
    client = ToolClient(
        sessions={
            "geo": mock_source,
            "device": mock_source,
            "auth": mock_source,
            "default": mock_source,
        }
    )
    store = InMemoryStateStore()
    trace_store = InMemoryTraceStore()
    tools_cfg = {
        "tools": {
            "geo_ip_lookup": {"server": "geo", "level": "READ", "timeout_ms": 50, "params": {"ip": "state.network.session_ip"}},
            "device_fingerprint": {"server": "device", "level": "READ", "timeout_ms": 50, "params": {"device_id": "state.device.id"}},
            "auth_history": {"server": "auth", "level": "READ", "timeout_ms": 50, "params": {"user_id": "state.account.user_id"}},
        }
    }
    tool_runner = ToolRunner(client=client, tools_cfg=tools_cfg, store=store)

    runner = CaseRunner(
        agent=adapter,
        tool_runner=tool_runner,
        store=store,
        trace_store=trace_store,
        model_tools=model.tools,
        tools_cfg=tools_cfg,
        stop_cfg=StopConfig(max_rounds=2),
    )

    case = Case(
        ticket_id="TCK-E2E-MODEL",
        type_token="<TYPE_PROMO>",
        text="Investigate alert",
        state={
            "network": {"session_ip": "103.2.4.5"},
            "device": {"id": "dev_1"},
            "account": {"user_id": "usr_1"},
        },
    )
    store.save_case(case)

    t0 = time.monotonic()
    payload = asyncio.run(runner.run_case("TCK-E2E-MODEL"))
    elapsed_ms = (time.monotonic() - t0) * 1000.0

    assert payload.outcome in ("AUTO_VERDICT", "ESCALATE")
    assert store.get_status("TCK-E2E-MODEL") in ("DECIDED", "ESCALATED")
    print(f"End-to-end run with un-trained Model A: {elapsed_ms:.2f}ms, Outcome: {payload.outcome}")


if __name__ == "__main__":
    test_rex_adapter_parity_with_investigator_agent()
    test_rex_adapter_end_to_end_with_case_runner_and_latency()
    print("PASS: test_orch_rex_adapter")
