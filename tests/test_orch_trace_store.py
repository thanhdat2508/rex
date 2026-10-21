"""Unit tests for TraceStore, PII masking, and telemetry validation (Step B9)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rex.orch.trace_store import InMemoryTraceStore, TraceStep, mask_pii


def test_mask_pii_sanitization():
    raw_params = {
        "ip": "103.28.32.11",
        "card_no": "4111123456789012",
        "email": "customer@bank.com",
        "safe_int": 42,
    }
    masked = mask_pii(raw_params)
    assert masked["ip"] == "103.x.x.11"
    assert masked["card_no"] == "4111****9012"
    assert masked["email"] == "c***@bank.com"
    assert masked["safe_int"] == 42


def test_in_memory_trace_store_lifecycle_and_schema():
    store = InMemoryTraceStore()
    ticket_id = "TCK-ANH-A-001"
    trace_id = store.start_trace(ticket_id)
    assert trace_id.startswith("rex-trace-")

    # Step 1: Tool call
    store.append_step(
        ticket_id,
        TraceStep(
            step_no=1,
            round=1,
            action="CALL_TOOL",
            status="OK",
            tool="geo_ip_lookup",
            tool_impact="READ",
            params={"ip": "103.28.32.11"},
            result={"country_mismatch": True, "ip_distance_km": 9200.0},
            sufficiency_before=0.40,
            sufficiency_after=0.62,
            duration_ms=41.5,
        ),
    )

    # Step 2: Final decision
    trace = store.finish_trace(ticket_id, final_action="AUTO_VERDICT", final_reason=None)
    trace_dict = trace.to_dict()

    assert trace_dict["trace_id"] == trace_id
    assert trace_dict["ticket_id"] == ticket_id
    assert trace_dict["final_action"] == "AUTO_VERDICT"
    assert len(trace_dict["steps"]) == 1

    s1 = trace_dict["steps"][0]
    assert s1["step_no"] == 1
    assert s1["tool"] == "geo_ip_lookup"
    assert s1["params"]["ip"] == "103.x.x.11"  # verify PII masked
    assert s1["sufficiency_before"] == 0.40
    assert s1["sufficiency_after"] == 0.62


if __name__ == "__main__":
    test_mask_pii_sanitization()
    test_in_memory_trace_store_lifecycle_and_schema()
    print("PASS: test_orch_trace_store")
