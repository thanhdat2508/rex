"""Unit tests for ToolRunner and replay of 'vụ anh A' through mock sources (Step B5)."""
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rex.orch.agent_port import Case
from rex.orch.fake_agent import FakeAgent
from rex.orch.registry import build_tool_specs, load_tools_config
from rex.orch.state_store import InMemoryStateStore
from rex.orch.tool_client import ToolClient
from rex.orch.tool_runner import ToolRunner
from rex.rex_registry import load_registry
from tests.orch_fake_sources import FakeMCPSources


def test_tool_runner_case_anh_a_replay():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reg_path = os.path.join(root, "examples", "rex", "registry.example.json")
    cfg_path = os.path.join(root, "config", "rex_tools.yaml")

    registry = load_registry(reg_path)
    cfg = load_tools_config(cfg_path)
    tool_specs_list = build_tool_specs(registry, cfg, registry.tools)
    tool_specs_dict = {s.name: s for s in tool_specs_list}

    # Setup mock MCP sources and client
    mock_source = FakeMCPSources(artificial_latency_s=0.015)
    client = ToolClient(
        sessions={
            "geo": mock_source,
            "device": mock_source,
            "auth": mock_source,
            "network": mock_source,
            "payment": mock_source,
            "behavior": mock_source,
        },
        tool_specs=tool_specs_dict,
        metric_specs=registry.metrics,
    )

    store = InMemoryStateStore()
    runner = ToolRunner(client=client, tools_cfg=cfg, store=store)

    agent = FakeAgent.case_anh_a()
    case = Case(
        ticket_id="TCK-ANH-A-001",
        type_token="<TYPE_PAYMENT_FRAUD>",
        text="Account takeover and anomalous geolocation login",
        state={
            "network": {"session_ip": "103.28.32.11"},
            "device": {"id": "dev_canvas_4401"},
            "account": {"user_id": "usr_99812"},
        },
    )
    store.save_case(case)

    async def execute():
        accumulated_results = {}

        # Round 1
        cmd1 = agent.start(case, available={})
        assert cmd1.kind == "CALL_TOOL"
        assert cmd1.tools == ["geo_ip_lookup", "device_fingerprint"]

        t0 = time.monotonic()
        r1_results = await runner.run(
            tools=cmd1.tools,
            round_k=1,
            deadline=time.monotonic() + 1.0,
            case=case,
            accumulated_results=accumulated_results,
        )
        round_1_ms = (time.monotonic() - t0) * 1000.0

        assert len(r1_results) == 2
        for r in r1_results:
            assert r.status == "OK"
            accumulated_results[r.tool] = r.metrics

        assert "country_mismatch" in accumulated_results["geo_ip_lookup"]
        assert accumulated_results["geo_ip_lookup"]["country_mismatch"] is True
        assert accumulated_results["device_fingerprint"]["unique_device_ratio"] == 0.25

        # Round 2
        cmd2 = agent.observe(
            text="geo_ip_lookup OK country_mismatch=1 | device_fingerprint OK unique_device_ratio=0.25",
            observations=accumulated_results,
            available={},
        )
        assert cmd2.kind == "CALL_TOOL"
        assert cmd2.tools == ["auth_history"]

        t1 = time.monotonic()
        r2_results = await runner.run(
            tools=cmd2.tools,
            round_k=2,
            deadline=time.monotonic() + 1.0,
            case=case,
            accumulated_results=accumulated_results,
        )
        round_2_ms = (time.monotonic() - t1) * 1000.0

        assert len(r2_results) == 1
        assert r2_results[0].status == "OK"
        accumulated_results[r2_results[0].tool] = r2_results[0].metrics
        assert accumulated_results["auth_history"]["password_reset_recent"] is True

        # Round 3
        cmd3 = agent.observe(
            text="auth_history OK failed_login_burst=5",
            observations=accumulated_results,
            available={},
        )
        assert cmd3.kind == "VERDICT"
        assert cmd3.verdict_probs["H_ato"] == 0.88

        # Verify StateStore contents
        called_tools = store.get_called_tools("TCK-ANH-A-001")
        assert called_tools == ["geo_ip_lookup", "device_fingerprint", "auth_history"]
        assert len(store.get_tool_results("TCK-ANH-A-001")) == 3

        print(f"Latency Round 1 (2 tools parallel): {round_1_ms:.2f}ms")
        print(f"Latency Round 2 (1 tool): {round_2_ms:.2f}ms")

    asyncio.run(execute())


if __name__ == "__main__":
    test_tool_runner_case_anh_a_replay()
    print("PASS: test_orch_tool_runner")
