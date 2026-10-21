"""Unit and integration tests for CaseRunner lifecycle and escalation pathways (Steps B10, B11)."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rex.orch.agent_port import AgentCommand, Case, CaseStatus
from rex.orch.fake_agent import FakeAgent
from rex.orch.guard import Guard
from rex.orch.registry import build_tool_specs, load_tools_config
from rex.orch.runner import CaseRunner
from rex.orch.state_store import InMemoryStateStore
from rex.orch.stop import StopConfig
from rex.orch.tool_client import ToolClient
from rex.orch.tool_runner import ToolRunner
from rex.orch.trace_store import InMemoryTraceStore
from rex.rex_registry import load_registry
from tests.orch_fake_sources import FakeMCPSources


def setup_runner_env(agent, stop_cfg=None, guard=None):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reg_path = os.path.join(root, "examples", "rex", "registry.example.json")
    cfg_path = os.path.join(root, "config", "rex_tools.yaml")

    registry = load_registry(reg_path)
    cfg = load_tools_config(cfg_path)
    tool_specs_list = build_tool_specs(registry, cfg, registry.tools)
    tool_specs_dict = {s.name: s for s in tool_specs_list}

    mock_source = FakeMCPSources(artificial_latency_s=0.005)
    client = ToolClient(
        sessions={k: mock_source for k in ["geo", "device", "auth", "network", "payment", "behavior"]},
        tool_specs=tool_specs_dict,
        metric_specs=registry.metrics,
    )
    store = InMemoryStateStore()
    trace_store = InMemoryTraceStore()
    guard = guard or Guard(allowed_levels={"READ"})
    tool_runner = ToolRunner(client=client, tools_cfg=cfg, store=store, guard=guard)

    runner = CaseRunner(
        agent=agent,
        tool_runner=tool_runner,
        store=store,
        trace_store=trace_store,
        model_tools=registry.tools,
        tools_cfg=cfg,
        stop_cfg=stop_cfg or StopConfig(),
        guard=guard,
    )
    return runner, store, trace_store


def test_runner_case_anh_a_auto_verdict():
    agent = FakeAgent.case_anh_a()
    runner, store, trace_store = setup_runner_env(agent)

    case = Case(
        ticket_id="TCK-RUNNER-001",
        type_token="<TYPE_PAYMENT_FRAUD>",
        state={
            "network": {"session_ip": "103.28.32.11"},
            "device": {"id": "dev_canvas_4401"},
            "account": {"user_id": "usr_99812"},
        },
    )
    store.save_case(case)

    payload = asyncio.run(runner.run_case("TCK-RUNNER-001"))

    assert payload.outcome == "AUTO_VERDICT"
    assert payload.verdict == "H_ato"
    assert payload.request_id == "TCK-RUNNER-001:1"
    assert store.get_status("TCK-RUNNER-001") == CaseStatus.DECIDED
    assert len(store.get_tool_results("TCK-RUNNER-001")) == 3

    trace = trace_store.get_trace("TCK-RUNNER-001")
    assert trace is not None
    assert trace.final_action == "AUTO_VERDICT"
    assert len(trace.steps) == 3


def test_runner_escalation_high_risk():
    agent = FakeAgent([
        AgentCommand(kind="CALL_TOOL", tools=["geo_ip_lookup"], s=0.50, risk=0.45)
    ])
    runner, store, _ = setup_runner_env(agent)

    case = Case(
        ticket_id="TCK-HIGH-RISK",
        type_token="<TYPE_PAYMENT_FRAUD>",
        state={"network": {"session_ip": "1.2.3.4"}},
    )
    store.save_case(case)

    # When risk >= 0.30 and sufficiency is high, or when risk threshold is tripped
    # Agent immediately proposes high risk
    agent_verdict = FakeAgent([
        AgentCommand(kind="VERDICT", s=0.90, risk=0.50, verdict_probs={"H_ato": 0.90})
    ])
    runner_risk, store_risk, _ = setup_runner_env(agent_verdict)
    store_risk.save_case(case)

    payload = asyncio.run(runner_risk.run_case("TCK-HIGH-RISK"))
    assert payload.outcome == "ESCALATE"
    assert payload.escalation_reason == "HIGH_RISK"
    assert payload.verdict is None
    assert store_risk.get_status("TCK-HIGH-RISK") == CaseStatus.ESCALATED


def test_runner_escalation_deadlock_stall():
    agent = FakeAgent([
        AgentCommand(kind="STALL", s=0.40, risk=0.10)
    ])
    runner, store, _ = setup_runner_env(agent)

    case = Case(ticket_id="TCK-STALL", type_token="<TYPE_PAYMENT_FRAUD>")
    store.save_case(case)

    payload = asyncio.run(runner.run_case("TCK-STALL"))
    assert payload.outcome == "ESCALATE"
    assert payload.escalation_reason == "DEADLOCK"
    assert payload.detail == "STALL_SIGNAL"
    assert payload.verdict is None
    assert store.get_status("TCK-STALL") == CaseStatus.ESCALATED


def test_runner_escalation_budget_exhausted():
    # Calling distinct tools each round until max_rounds is reached
    agent = FakeAgent([
        AgentCommand(kind="CALL_TOOL", tools=["geo_ip_lookup"], s=0.30, risk=0.10),
        AgentCommand(kind="CALL_TOOL", tools=["device_fingerprint"], s=0.50, risk=0.10),
        AgentCommand(kind="CALL_TOOL", tools=["auth_history"], s=0.70, risk=0.10),
        AgentCommand(kind="CALL_TOOL", tools=["ip_reputation"], s=0.80, risk=0.10),
    ])
    runner, store, _ = setup_runner_env(agent, stop_cfg=StopConfig(max_rounds=3))

    case = Case(
        ticket_id="TCK-BUDGET",
        type_token="<TYPE_PAYMENT_FRAUD>",
        state={
            "network": {"session_ip": "1.2.3.4"},
            "device": {"id": "dev_123"},
            "account": {"user_id": "usr_123"},
        },
    )
    store.save_case(case)

    payload = asyncio.run(runner.run_case("TCK-BUDGET"))
    assert payload.outcome == "ESCALATE"
    assert payload.escalation_reason == "BUDGET_EXHAUSTED"
    assert payload.verdict is None
    assert store.get_status("TCK-BUDGET") == CaseStatus.ESCALATED


if __name__ == "__main__":
    test_runner_case_anh_a_auto_verdict()
    test_runner_escalation_high_risk()
    test_runner_escalation_deadlock_stall()
    test_runner_escalation_budget_exhausted()
    print("PASS: test_orch_runner")
