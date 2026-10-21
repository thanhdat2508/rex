"""Unit tests for SqliteStateStore, atomic lease locking, and outbox delivery (Step B14)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rex.orch.agent_port import Case, CaseStatus
from rex.orch.backends.sqlite_store import SqliteStateStore
from rex.orch.state_store import CaseOutcome, StateStore, ToolResult


def test_sqlite_state_store_implements_protocol():
    store = SqliteStateStore(":memory:")
    assert isinstance(store, StateStore)


def test_atomic_lease_contention_and_expiration():
    store = SqliteStateStore(":memory:")
    case = Case(ticket_id="TCK-LEASE-1", type_token="<FRAUD>")
    store.save_case(case)

    # Worker A acquires lease
    assert store.acquire_lease("TCK-LEASE-1", worker_id="worker_A", lease_duration_s=0.2) is True

    # Worker B tries to acquire while lease is active -> fails
    assert store.acquire_lease("TCK-LEASE-1", worker_id="worker_B", lease_duration_s=0.2) is False

    # Worker A releases lease -> Worker B can now acquire
    assert store.release_lease("TCK-LEASE-1", worker_id="worker_A") is True
    assert store.acquire_lease("TCK-LEASE-1", worker_id="worker_B", lease_duration_s=0.05) is True

    # Worker B's lease expires after 50ms -> Worker C can acquire expired lease
    time.sleep(0.08)
    assert store.acquire_lease("TCK-LEASE-1", worker_id="worker_C", lease_duration_s=1.0) is True


def test_transactional_outbox_atomicity():
    store = SqliteStateStore(":memory:")
    case = Case(ticket_id="TCK-OUTBOX-1", type_token="<FRAUD>")
    store.save_case(case)
    store.set_status("TCK-OUTBOX-1", CaseStatus.INVESTIGATING)

    outcome = CaseOutcome(
        action="AUTO_VERDICT",
        verdict="H_ato",
        verdict_probs={"H_ato": 0.88},
        conformal_set=["H_ato"],
        risk=0.15,
        trace_id="rex-trace-1234",
    )
    seq = store.record_outcome("TCK-OUTBOX-1", outcome)
    assert seq == 1

    pending = store.get_outbox_pending()
    assert len(pending) == 1
    item = pending[0]
    assert item["ticket_id"] == "TCK-OUTBOX-1"
    assert item["decision_seq"] == 1
    assert item["payload"]["action"] == "AUTO_VERDICT"
    assert item["payload"]["verdict"] == "H_ato"
    assert item["payload"]["request_id"] == "TCK-OUTBOX-1:1"

    # Mark delivered
    store.mark_outbox_delivered(item["id"])
    assert len(store.get_outbox_pending()) == 0


def test_evidence_and_tool_persistence_across_connections():
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    try:
        store1 = SqliteStateStore(path)
        case = Case(ticket_id="TCK-PERSIST", type_token="<FRAUD>")
        store1.save_case(case)
        store1.append_evidence("TCK-PERSIST", 1, "evidence from round 1")
        store1.append_tool_result("TCK-PERSIST", ToolResult(tool="geo_ip_lookup", status="OK", metrics={"ip_dist": 100}))

        # Reopen from another store connection
        store2 = SqliteStateStore(path)
        assert store2.get_evidence_chain("TCK-PERSIST") == ["evidence from round 1"]
        assert store2.get_called_tools("TCK-PERSIST") == ["geo_ip_lookup"]
        store1.close()
        store2.close()
    finally:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


if __name__ == "__main__":
    test_sqlite_state_store_implements_protocol()
    test_atomic_lease_contention_and_expiration()
    test_transactional_outbox_atomicity()
    test_evidence_and_tool_persistence_across_connections()
    print("PASS: test_orch_backends")
