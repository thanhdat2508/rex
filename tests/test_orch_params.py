"""Unit tests for parameter extraction and availability masking (Step B3)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rex.orch.agent_port import Case
from rex.orch.params import compute_availability, resolve_param, resolve_tool_args


def test_resolve_param_from_state():
    case = Case(
        ticket_id="TCK-1",
        type_token="<FRAUD>",
        state={
            "network": {"session_ip": "103.21.244.2"},
            "account": {"user_id": "usr_992"},
        },
    )
    ip = resolve_param("state.network.session_ip", case, {})
    assert ip == "103.21.244.2"

    uid = resolve_param("state.account.user_id", case, {})
    assert uid == "usr_992"

    missing = resolve_param("state.device.id", case, {})
    assert missing is None


def test_resolve_param_from_results():
    case = Case(ticket_id="TCK-1", type_token="<FRAUD>")
    results = {"geo_ip_lookup": {"country_mismatch": 1, "ip_distance_km": 9200.0}}

    val = resolve_param("results.geo_ip_lookup.country_mismatch", case, results)
    assert val == 1

    missing_tool = resolve_param("results.auth_history.failed_logins", case, results)
    assert missing_tool is None


def test_resolve_tool_args_all_present():
    cfg = {
        "tools": {
            "geo_ip_lookup": {
                "params": {"ip": "state.network.session_ip"},
            },
            "dependent_tool": {
                "params": {
                    "ip": "state.network.session_ip",
                    "dist": "results.geo_ip_lookup.ip_distance_km",
                },
            },
        }
    }
    case = Case(
        ticket_id="TCK-1",
        type_token="<FRAUD>",
        state={"network": {"session_ip": "1.2.3.4"}},
    )

    args = resolve_tool_args("geo_ip_lookup", cfg, case, {})
    assert args == {"ip": "1.2.3.4"}

    # dependent_tool lacks 'dist' before geo_ip_lookup runs
    args_dep_before = resolve_tool_args("dependent_tool", cfg, case, {})
    assert args_dep_before is None

    # after geo_ip_lookup produces ip_distance_km
    results = {"geo_ip_lookup": {"ip_distance_km": 500.0}}
    args_dep_after = resolve_tool_args("dependent_tool", cfg, case, results)
    assert args_dep_after == {"ip": "1.2.3.4", "dist": 500.0}


def test_compute_availability_with_guard_and_params():
    cfg = {
        "tools": {
            "geo_ip_lookup": {"params": {"ip": "state.network.session_ip"}},
            "device_fingerprint": {"params": {"dev_id": "state.device.id"}},
        }
    }
    case = Case(
        ticket_id="TCK-1",
        type_token="<FRAUD>",
        state={"network": {"session_ip": "1.2.3.4"}},  # missing device.id
    )

    avail = compute_availability(["geo_ip_lookup", "device_fingerprint"], cfg, case, {})
    assert avail["geo_ip_lookup"] is True
    assert avail["device_fingerprint"] is False

    # Mock guard that suppresses geo_ip_lookup
    class MockGuard:
        def is_allowed(self, tool_name: str) -> bool:
            return tool_name != "geo_ip_lookup"

    avail_guarded = compute_availability(
        ["geo_ip_lookup", "device_fingerprint"], cfg, case, {}, guard=MockGuard()
    )
    assert avail_guarded["geo_ip_lookup"] is False
    assert avail_guarded["device_fingerprint"] is False


if __name__ == "__main__":
    test_resolve_param_from_state()
    test_resolve_param_from_results()
    test_resolve_tool_args_all_present()
    test_compute_availability_with_guard_and_params()
    print("PASS: test_orch_params")
