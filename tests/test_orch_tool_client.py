"""Unit tests for ToolClient concurrency, timeout isolation, and metric validation (Step B4)."""
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from rex.orch.tool_client import ToolCall, ToolClient, ToolSession, validate_metrics
from rex.rex_registry import MetricSpec, ToolSpec


class MockSession(ToolSession):
    def __init__(self, responses: dict, delays: dict = None, failures: set = None):
        self.responses = responses
        self.delays = delays or {}
        self.failures = failures or set()

    async def call_tool(self, remote_name: str, args: dict, timeout_s: float) -> dict:
        delay = self.delays.get(remote_name, 0.0)
        if delay > 0:
            await asyncio.sleep(delay)
        if remote_name in self.failures:
            raise RuntimeError(f"Simulated failure on {remote_name}")
        return self.responses.get(remote_name, {})


def test_parallel_execution_takes_max_not_sum():
    async def run():
        session = MockSession(
            responses={
                "geo_ip_lookup": {"country_mismatch": True, "ip_distance_km": 100.0},
                "device_fingerprint": {"unique_device_ratio": 0.5},
            },
            delays={
                "geo_ip_lookup": 0.05,        # 50ms
                "device_fingerprint": 0.04,   # 40ms
            },
        )
        client = ToolClient(sessions={"geo": session, "device": session})
        calls = [
            ToolCall(tool="geo_ip_lookup", server="geo", remote_name="geo_ip_lookup", args={}, timeout_s=0.2),
            ToolCall(tool="device_fingerprint", server="device", remote_name="device_fingerprint", args={}, timeout_s=0.2),
        ]

        t0 = time.monotonic()
        results = await client.call_tools(calls, deadline=t0 + 1.0)
        elapsed = time.monotonic() - t0

        assert len(results) == 2
        assert results[0].status == "OK"
        assert results[1].status == "OK"
        # Total elapsed time should be approximately ~50ms (< 85ms), definitely not serialized 90ms+
        assert elapsed < 0.085, f"Expected concurrent execution, elapsed: {elapsed:.3f}s"

    asyncio.run(run())


def test_timeout_and_error_isolation():
    async def run():
        session = MockSession(
            responses={
                "fast_tool": {"val": 10},
                "slow_tool": {"val": 99},
            },
            delays={
                "fast_tool": 0.01,
                "slow_tool": 0.50,            # 500ms delay, exceeds 50ms timeout
            },
            failures={"broken_tool"},
        )
        client = ToolClient(sessions={"srv": session})
        calls = [
            ToolCall(tool="fast_tool", server="srv", remote_name="fast_tool", args={}, timeout_s=0.2),
            ToolCall(tool="slow_tool", server="srv", remote_name="slow_tool", args={}, timeout_s=0.04),
            ToolCall(tool="broken_tool", server="srv", remote_name="broken_tool", args={}, timeout_s=0.2),
        ]

        deadline = time.monotonic() + 1.0
        results = await client.call_tools(calls, deadline)

        res_map = {r.tool: r for r in results}
        assert res_map["fast_tool"].status == "OK"
        assert res_map["fast_tool"].metrics == {"val": 10}

        assert res_map["slow_tool"].status == "TIMEOUT"
        assert len(res_map["slow_tool"].errors) > 0

        assert res_map["broken_tool"].status == "ERROR"
        assert any("RuntimeError" in e for e in res_map["broken_tool"].errors)

    asyncio.run(run())


def test_metric_validation_bounds_and_filtering():
    tool_spec = ToolSpec(
        name="geo_ip_lookup",
        returns=["country_mismatch", "ip_distance_km", "valid_ratio"],
        level="READ",
    )
    metric_specs = {
        "country_mismatch": MetricSpec(name="country_mismatch", kind="bool"),
        "ip_distance_km": MetricSpec(name="ip_distance_km", kind="float", min_val=0.0, max_val=20000.0),
        "valid_ratio": MetricSpec(name="valid_ratio", kind="float", min_val=0.0, max_val=1.0),
    }

    raw = {
        "country_mismatch": 1,                # valid bool representation
        "ip_distance_km": 25000.0,            # exceeds max_val 20000.0 -> discarded
        "valid_ratio": 0.75,                  # valid
        "extra_unwanted": 999,                # not in returns -> discarded
    }

    metrics, errors = validate_metrics("geo_ip_lookup", raw, tool_spec, metric_specs)
    assert metrics == {"country_mismatch": True, "valid_ratio": 0.75}
    assert any("max_val" in e for e in errors)


def test_heavy_tool_is_filtered_never_dispatched():
    async def run():
        session = MockSession(responses={"block_card": {"blocked": True}})
        client = ToolClient(sessions={"pay": session})

        calls = [
            ToolCall(tool="block_card", server="pay", remote_name="block_card", args={}, timeout_s=0.1, level="HEAVY")
        ]
        results = await client.call_tools(calls, deadline=time.monotonic() + 1.0)
        assert len(results) == 1
        assert results[0].status == "FILTERED"
        assert "HEAVY" in results[0].errors[0]

    asyncio.run(run())


if __name__ == "__main__":
    test_parallel_execution_takes_max_not_sum()
    test_timeout_and_error_isolation()
    test_metric_validation_bounds_and_filtering()
    test_heavy_tool_is_filtered_never_dispatched()
    print("PASS: test_orch_tool_client")
