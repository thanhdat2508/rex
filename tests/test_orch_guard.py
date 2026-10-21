"""Unit tests for Guard security checks and post-forward tool filtering (Step B6)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rex.orch.guard import Guard


def test_guard_pre_forward_levels():
    guard = Guard(allowed_levels={"READ"})
    assert guard.is_allowed("geo_ip_lookup", level="READ") is True
    assert guard.is_allowed("block_card", level="HEAVY") is False
    assert guard.is_allowed("send_otp", level="LIGHT") is False


def test_guard_high_risk_mode_strictly_read():
    guard = Guard(allowed_levels={"READ", "LIGHT"}, high_risk_mode=True)
    assert guard.is_allowed("geo_ip_lookup", level="READ") is True
    assert guard.is_allowed("send_otp", level="LIGHT") is False


def test_guard_post_forward_filtering_prohibited_level():
    guard = Guard(allowed_levels={"READ"})
    cfg = {
        "tools": {
            "block_card": {"level": "HEAVY"},
        }
    }
    approved, reason, detail = guard.filter_selected_tools(
        selected_tools=["block_card"],
        tools_cfg=cfg,
        available_mask={"block_card": True},
    )
    assert approved == []
    assert reason == "TOOL_BLOCKED"
    assert detail == "PROHIBITED_LEVEL"


def test_guard_post_forward_filtering_missing_params():
    guard = Guard(allowed_levels={"READ"})
    cfg = {
        "tools": {
            "geo_ip_lookup": {"level": "READ"},
        }
    }
    # geo_ip_lookup was selected by model, but params are missing in available_mask
    approved, reason, detail = guard.filter_selected_tools(
        selected_tools=["geo_ip_lookup"],
        tools_cfg=cfg,
        available_mask={"geo_ip_lookup": False},
    )
    assert approved == []
    assert reason == "DEADLOCK"
    assert detail == "NO_CALLABLE_TOOL"


def test_guard_post_forward_filtering_anti_repetition():
    guard = Guard(allowed_levels={"READ"})
    cfg = {
        "tools": {
            "geo_ip_lookup": {"level": "READ"},
            "device_fingerprint": {"level": "READ"},
        }
    }
    # geo_ip_lookup was already called in round 1
    approved, reason, detail = guard.filter_selected_tools(
        selected_tools=["geo_ip_lookup", "device_fingerprint"],
        tools_cfg=cfg,
        available_mask={"geo_ip_lookup": True, "device_fingerprint": True},
        called_tools=["geo_ip_lookup"],
    )
    assert approved == ["device_fingerprint"]
    assert reason is None


if __name__ == "__main__":
    test_guard_pre_forward_levels()
    test_guard_high_risk_mode_strictly_read()
    test_guard_post_forward_filtering_prohibited_level()
    test_guard_post_forward_filtering_missing_params()
    test_guard_post_forward_filtering_anti_repetition()
    print("PASS: test_orch_guard")
