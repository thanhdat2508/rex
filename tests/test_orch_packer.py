"""Unit tests for Evidence Packer and 64-token boundary compliance (Step B7)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rex.orch.packer import format_metric_value, format_tool_result, pack_evidence
from rex.orch.state_store import ToolResult


def test_format_metric_value():
    assert format_metric_value(True) == "1"
    assert format_metric_value(False) == "0"
    assert format_metric_value(9200.0) == "9200"
    assert format_metric_value(0.25) == "0.25"
    assert format_metric_value("short_str") == "short_str"
    # Long string capped at 32 chars
    long_str = "a" * 50
    formatted = format_metric_value(long_str)
    assert len(formatted) <= 32
    assert formatted.endswith("...")


def test_format_tool_result_ok_and_timeout():
    res_ok = ToolResult(
        tool="geo_ip_lookup",
        status="OK",
        metrics={"country_mismatch": True, "ip_distance_km": 9200.0},
    )
    res_timeout = ToolResult(
        tool="auth_history",
        status="TIMEOUT",
    )

    str_ok = format_tool_result(res_ok)
    assert str_ok == "geo_ip_lookup OK country_mismatch=1 ip_distance_km=9200"

    str_timeout = format_tool_result(res_timeout)
    assert str_timeout == "auth_history TIMEOUT"


def test_pack_evidence_multi_tool_round():
    results = [
        ToolResult(
            tool="geo_ip_lookup",
            status="OK",
            metrics={"country_mismatch": True, "ip_distance_km": 9200.0},
        ),
        ToolResult(
            tool="device_fingerprint",
            status="OK",
            metrics={"device_seen_before": False, "unique_device_ratio": 0.25},
        ),
        ToolResult(
            tool="auth_history",
            status="TIMEOUT",
        ),
    ]

    packed = pack_evidence(results, round_k=1, max_tokens=64)
    expected_text = (
        "geo_ip_lookup OK country_mismatch=1 ip_distance_km=9200 | "
        "device_fingerprint OK device_seen_before=0 unique_device_ratio=0.25 | "
        "auth_history TIMEOUT"
    )
    assert packed.text == expected_text
    assert not packed.is_truncated
    assert "geo_ip_lookup" in packed.observations
    assert "device_fingerprint" in packed.observations
    assert "auth_history" not in packed.observations


def test_pack_evidence_truncation_detection():
    results = [
        ToolResult(tool=f"tool_{i}", status="OK", metrics={"val": i * 100})
        for i in range(20)
    ]
    packed = pack_evidence(results, round_k=1, max_tokens=10)
    assert packed.is_truncated is True


if __name__ == "__main__":
    test_format_metric_value()
    test_format_tool_result_ok_and_timeout()
    test_pack_evidence_multi_tool_round()
    test_pack_evidence_truncation_detection()
    print("PASS: test_orch_packer")
