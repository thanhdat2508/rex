"""Unit tests for StateStore and InMemoryStateStore lifecycle (Step B2)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from rex.orch.agent_port import Case, CaseStatus
from rex.orch.state_store import (
    CaseOutcome,
    InMemoryStateStore,
    InvalidStateTransitionError,
    StateStore,
    ToolResult,
)


def test_in_memory_state_store_implements_protocol():
    store = InMemoryStateStore()
    assert isinstance(store, StateStore)


def test_valid_state_transitions():
    store = InMemoryStateStore()
    case = Case(ticket_id="TCK-1", type_token="<FRAUD>")
    store.save_case(case)

    assert store.get_status("TCK-1") == CaseStatus.OPEN

    store.set_status("TCK-1", CaseStatus.INVESTIGATING)
    assert store.get_status("TCK-1") == CaseStatus.INVESTIGATING

    store.set_status("TCK-1", CaseStatus.DECIDED)
    assert store.get_status("TCK-1") == CaseStatus.DECIDED


def test_invalid_state_transitions_raise_error():
    store = InMemoryStateStore()
    case = Case(ticket_id="TCK-2", type_token="<FRAUD>")
    store.save_case(case)

    # Cannot jump from OPEN directly to DECIDED
    with pytest.raises(InvalidStateTransitionError):
        store.set_status("TCK-2", CaseStatus.DECIDED)

    # Transition to INVESTIGATING
    store.set_status("TCK-2", CaseStatus.INVESTIGATING)

    # Transition to ESCALATED
    store.set_status("TCK-2", CaseStatus.ESCALATED)

    # Cannot transition once terminal (ESCALATED -> INVESTIGATING)
    with pytest.raises(InvalidStateTransitionError):
        store.set_status("TCK-2", CaseStatus.INVESTIGATING)


def test_evidence_and_tool_results_chronology():
    store = InMemoryStateStore()
    case = Case(ticket_id="TCK-3", type_token="<FRAUD>")
    store.save_case(case)

    store.append_evidence("TCK-3", 1, "evidence round 1")
    store.append_evidence("TCK-3", 2, "evidence round 2")

    chain = store.get_evidence_chain("TCK-3")
    assert chain == ["evidence round 1", "evidence round 2"]

    store.append_tool_result("TCK-3", ToolResult(tool="geo_ip_lookup", status="OK", metrics={"country_mismatch": 1}))
    store.append_tool_result("TCK-3", ToolResult(tool="auth_history", status="TIMEOUT"))

    called = store.get_called_tools("TCK-3")
    assert called == ["geo_ip_lookup", "auth_history"]


def test_record_outcome_increments_decision_seq():
    store = InMemoryStateStore()
    case = Case(ticket_id="TCK-4", type_token="<FRAUD>")
    store.save_case(case)

    seq1 = store.record_outcome(
        "TCK-4",
        CaseOutcome(
            action="AUTO_VERDICT",
            verdict="H_ato",
            verdict_probs={"H_ato": 0.9},
            risk=0.1,
        ),
    )
    assert seq1 == 1

    outcome = store.get_outcome("TCK-4")
    assert outcome is not None
    assert outcome.action == "AUTO_VERDICT"
    assert outcome.verdict == "H_ato"


if __name__ == "__main__":
    test_in_memory_state_store_implements_protocol()
    test_valid_state_transitions()
    test_invalid_state_transitions_raise_error()
    test_evidence_and_tool_results_chronology()
    test_record_outcome_increments_decision_seq()
    print("PASS: test_orch_state_store")
