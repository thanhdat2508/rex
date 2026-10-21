#!/usr/bin/env python3
"""Benchmark orchestration round and end-to-end case latencies on CPU (Step B15)."""
import asyncio
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoConfig, AutoModel, BertTokenizer

from rex.orch.agent_port import Case
from rex.orch.metrics import OrchMetrics, compute_percentiles
from rex.orch.rex_adapter import RexAdapter
from rex.orch.runner import CaseRunner
from rex.orch.state_store import InMemoryStateStore
from rex.orch.stop import StopConfig
from rex.orch.tool_client import ToolClient
from rex.orch.tool_runner import ToolRunner
from rex.orch.trace_store import InMemoryTraceStore
from rex.rex_model import RexDecisionModel
from rex.rex_poe import ComparisonOperator, FalsificationRule
from rex.rex_vocab import add_type_tokens
from tests.orch_fake_sources import FakeMCPSources


def create_offline_tok() -> BertTokenizer:
    fd, vocab_path = tempfile.mkstemp(suffix=".txt", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(
            "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n"
            "which\nhypothesis\nexplains\nthe\nincident\n"
            "bot\nsybil\nshared\nnovel\nstate\nanomaly\n"
        )
    tok = BertTokenizer(vocab_path)
    os.remove(vocab_path)
    add_type_tokens(tok, None, {"<TYPE_PROMO>": "promo"})
    return tok


def create_model(d: int = 32) -> RexDecisionModel:
    cfg = AutoConfig.for_model(
        "bert",
        hidden_size=d,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        vocab_size=200,
    )
    enc = AutoModel.from_config(cfg)
    tools = ["geo_ip_lookup", "device_fingerprint"]
    return RexDecisionModel(
        encoder=enc,
        tools=tools,
        type_tokens={"<TYPE_PROMO>": "promo"},
        head_layers=1,
        n_act=2,
    )


async def main():
    print("=" * 60)
    print("Rex Orchestration Latency & Percentile Benchmark")
    print("=" * 60)

    tok = create_offline_tok()
    model = create_model()
    mock_source = FakeMCPSources(artificial_latency_s=0.003)
    client = ToolClient(sessions={"geo": mock_source, "device": mock_source, "default": mock_source})

    tools_cfg = {
        "tools": {
            "geo_ip_lookup": {"server": "geo", "level": "READ", "timeout_ms": 50, "params": {"ip": "state.network.session_ip"}},
            "device_fingerprint": {"server": "device", "level": "READ", "timeout_ms": 50, "params": {"device_id": "state.device.id"}},
        }
    }

    hypotheses = {"H_bot": "bot", "H_novel": "novel"}
    rules = {"H_bot": [FalsificationRule(metric="country_mismatch", operator=ComparisonOperator.EQ, threshold=True)]}
    tool_hyp_mapping = {"geo_ip_lookup": ["H_bot"], "device_fingerprint": ["H_novel"]}

    metrics = OrchMetrics()
    n_iterations = 20

    print(f"Running {n_iterations} end-to-end benchmark trials...")

    for i in range(n_iterations):
        store = InMemoryStateStore()
        trace_store = InMemoryTraceStore()
        tool_runner = ToolRunner(client=client, tools_cfg=tools_cfg, store=store)

        adapter = RexAdapter(
            model=model,
            tok=tok,
            instruction="Identify anomaly",
            hypotheses=hypotheses,
            rules=rules,
            tool_hyp_mapping=tool_hyp_mapping,
        )

        runner = CaseRunner(
            agent=adapter,
            tool_runner=tool_runner,
            store=store,
            trace_store=trace_store,
            model_tools=model.tools,
            tools_cfg=tools_cfg,
            stop_cfg=StopConfig(max_rounds=2),
        )

        ticket_id = f"TCK-BENCH-{i}"
        case = Case(
            ticket_id=ticket_id,
            type_token="<TYPE_PROMO>",
            text="High velocity login bursts",
            state={"network": {"session_ip": "1.2.3.4"}, "device": {"id": "d1"}},
        )
        store.save_case(case)

        t0 = time.monotonic()
        payload = await runner.run_case(ticket_id)
        case_ms = (time.monotonic() - t0) * 1000.0

        metrics.record_case_outcome(payload.outcome, payload.escalation_reason, duration_ms=case_ms)

    snap = metrics.snapshot()
    print("\n--- Latency Percentile Summary (ms) ---")
    print(f"Total Cases Run: {snap['cases_total']}")
    print(f"Escalation Rate: {snap['escalation_rate']:.1%}")
    case_lat = snap["latency_case_ms"]
    print(f"Case Latency: p50={case_lat['p50']}ms | p95={case_lat['p95']}ms | p99={case_lat['p99']}ms | avg={case_lat['avg']}ms")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
