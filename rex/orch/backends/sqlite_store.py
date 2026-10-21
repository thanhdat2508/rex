"""Persistent SQLite StateStore with atomic lease locking and transactional outbox (B14).

Provides persistent storage for incident cases, guarantees mutual exclusion via atomic leases,
and ensures exactly-once handoff semantics via an integrated outbox table.
"""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

from rex.orch.agent_port import Case, CaseStatus
from rex.orch.state_store import (
    CaseOutcome,
    InvalidStateTransitionError,
    RoundRecord,
    StateStore,
    ToolResult,
)


class SqliteStateStore(StateStore):
    """Production persistent backend implementing StateStore with leases and outbox."""

    VALID_TRANSITIONS = {
        CaseStatus.OPEN: {CaseStatus.INVESTIGATING},
        CaseStatus.INVESTIGATING: {CaseStatus.DECIDED, CaseStatus.ESCALATED},
        CaseStatus.DECIDED: set(),
        CaseStatus.ESCALATED: set(),
    }

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()

    def _init_tables(self) -> None:
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS cases (
                    ticket_id TEXT PRIMARY KEY,
                    type_token TEXT NOT NULL,
                    text TEXT,
                    state_json TEXT,
                    initial_evidence TEXT,
                    metadata_json TEXT,
                    status TEXT NOT NULL,
                    lease_owner TEXT,
                    lease_expires_at REAL DEFAULT 0,
                    decision_seq INTEGER DEFAULT 0,
                    created_at REAL,
                    updated_at REAL
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS tool_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket_id TEXT NOT NULL,
                    tool TEXT NOT NULL,
                    status TEXT NOT NULL,
                    metrics_json TEXT,
                    errors_json TEXT,
                    duration_ms REAL,
                    created_at REAL,
                    FOREIGN KEY(ticket_id) REFERENCES cases(ticket_id)
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS evidence_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket_id TEXT NOT NULL,
                    round_k INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    created_at REAL,
                    FOREIGN KEY(ticket_id) REFERENCES cases(ticket_id)
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS rounds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket_id TEXT NOT NULL,
                    round_k INTEGER NOT NULL,
                    s REAL NOT NULL,
                    n_calls INTEGER NOT NULL,
                    called_tools_json TEXT,
                    created_at REAL,
                    FOREIGN KEY(ticket_id) REFERENCES cases(ticket_id)
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS outbox (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket_id TEXT NOT NULL,
                    decision_seq INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT DEFAULT 'PENDING',
                    created_at REAL,
                    FOREIGN KEY(ticket_id) REFERENCES cases(ticket_id)
                )
            """)

    def save_case(self, case: Case) -> None:
        now = time.time()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO cases (
                    ticket_id, type_token, text, state_json, initial_evidence,
                    metadata_json, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticket_id) DO UPDATE SET
                    text=excluded.text,
                    state_json=excluded.state_json,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (
                    case.ticket_id,
                    case.type_token,
                    case.text,
                    json.dumps(case.state, ensure_ascii=False),
                    case.initial_evidence,
                    json.dumps(case.metadata, ensure_ascii=False),
                    CaseStatus.OPEN.value,
                    now,
                    now,
                ),
            )

    def load_case(self, ticket_id: str) -> Case:
        cursor = self.conn.execute(
            "SELECT ticket_id, type_token, text, state_json, initial_evidence, metadata_json FROM cases WHERE ticket_id = ?",
            (ticket_id,),
        )
        row = cursor.fetchone()
        if not row:
            raise KeyError(f"Case '{ticket_id}' not found in SqliteStateStore")
        return Case(
            ticket_id=row["ticket_id"],
            type_token=row["type_token"],
            text=row["text"] or "",
            state=json.loads(row["state_json"] or "{}"),
            initial_evidence=row["initial_evidence"] or "",
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    def get_status(self, ticket_id: str) -> CaseStatus:
        cursor = self.conn.execute("SELECT status FROM cases WHERE ticket_id = ?", (ticket_id,))
        row = cursor.fetchone()
        if not row:
            raise KeyError(f"Case '{ticket_id}' not found")
        return CaseStatus(row["status"])

    def set_status(self, ticket_id: str, status: CaseStatus) -> None:
        current = self.get_status(ticket_id)
        if current == status:
            return

        allowed = self.VALID_TRANSITIONS.get(current, set())
        if status not in allowed:
            raise InvalidStateTransitionError(
                f"Cannot transition case '{ticket_id}' from {current} to {status}. "
                f"Allowed: {allowed}"
            )

        with self.conn:
            self.conn.execute(
                "UPDATE cases SET status = ?, updated_at = ? WHERE ticket_id = ?",
                (status.value, time.time(), ticket_id),
            )

    def append_tool_result(self, ticket_id: str, result: ToolResult) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO tool_results (ticket_id, tool, status, metrics_json, errors_json, duration_ms, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticket_id,
                    result.tool,
                    result.status,
                    json.dumps(result.metrics, ensure_ascii=False),
                    json.dumps(result.errors, ensure_ascii=False),
                    result.duration_ms,
                    time.time(),
                ),
            )

    def append_evidence(self, ticket_id: str, round_k: int, text: str) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO evidence_records (ticket_id, round_k, text, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (ticket_id, round_k, text, time.time()),
            )

    def record_round(
        self,
        ticket_id: str,
        round_k: int,
        s: float,
        n_calls: int,
        called_tools: List[str],
    ) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO rounds (ticket_id, round_k, s, n_calls, called_tools_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (ticket_id, round_k, s, n_calls, json.dumps(called_tools), time.time()),
            )

    def record_outcome(self, ticket_id: str, outcome: CaseOutcome) -> int:
        now = time.time()
        with self.conn:
            cursor = self.conn.execute(
                "SELECT decision_seq FROM cases WHERE ticket_id = ?", (ticket_id,)
            )
            row = cursor.fetchone()
            if not row:
                raise KeyError(f"Case '{ticket_id}' not found")

            new_seq = (row["decision_seq"] or 0) + 1
            self.conn.execute(
                "UPDATE cases SET decision_seq = ?, updated_at = ? WHERE ticket_id = ?",
                (new_seq, now, ticket_id),
            )

            # Insert into transactional outbox
            payload = {
                "request_id": f"{ticket_id}:{new_seq}",
                "ticket_id": ticket_id,
                "decision_seq": new_seq,
                "action": outcome.action,
                "verdict": outcome.verdict,
                "verdict_probs": outcome.verdict_probs,
                "conformal_set": outcome.conformal_set,
                "risk": outcome.risk,
                "escalation_reason": outcome.escalation_reason,
                "detail": outcome.detail,
                "trace_id": outcome.trace_id,
                "decided_at": outcome.decided_at,
            }
            self.conn.execute(
                """
                INSERT INTO outbox (ticket_id, decision_seq, payload_json, status, created_at)
                VALUES (?, ?, ?, 'PENDING', ?)
                """,
                (ticket_id, new_seq, json.dumps(payload, ensure_ascii=False), now),
            )
            return new_seq

    def get_evidence_chain(self, ticket_id: str) -> List[str]:
        cursor = self.conn.execute(
            "SELECT text FROM evidence_records WHERE ticket_id = ? ORDER BY id ASC",
            (ticket_id,),
        )
        return [row["text"] for row in cursor.fetchall()]

    def get_called_tools(self, ticket_id: str) -> List[str]:
        cursor = self.conn.execute(
            "SELECT DISTINCT tool FROM tool_results WHERE ticket_id = ? ORDER BY id ASC",
            (ticket_id,),
        )
        return [row["tool"] for row in cursor.fetchall()]

    # --- Atomic Lease Locking Methods ---

    def acquire_lease(
        self,
        ticket_id: str,
        worker_id: str,
        lease_duration_s: float = 60.0,
    ) -> bool:
        """Attempt to atomically acquire or renew a case lease."""
        now = time.time()
        expires_at = now + lease_duration_s
        with self.conn:
            cursor = self.conn.execute(
                """
                UPDATE cases
                SET lease_owner = ?, lease_expires_at = ?, updated_at = ?
                WHERE ticket_id = ?
                  AND (lease_owner IS NULL OR lease_expires_at < ? OR lease_owner = ?)
                """,
                (worker_id, expires_at, now, ticket_id, now, worker_id),
            )
            return cursor.rowcount == 1

    def release_lease(self, ticket_id: str, worker_id: str) -> bool:
        with self.conn:
            cursor = self.conn.execute(
                """
                UPDATE cases
                SET lease_owner = NULL, lease_expires_at = 0, updated_at = ?
                WHERE ticket_id = ? AND lease_owner = ?
                """,
                (time.time(), ticket_id, worker_id),
            )
            return cursor.rowcount == 1

    def get_outbox_pending(self) -> List[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT id, ticket_id, decision_seq, payload_json FROM outbox WHERE status = 'PENDING' ORDER BY id ASC"
        )
        items = []
        for row in cursor.fetchall():
            items.append({
                "id": row["id"],
                "ticket_id": row["ticket_id"],
                "decision_seq": row["decision_seq"],
                "payload": json.loads(row["payload_json"]),
            })
        return items

    def mark_outbox_delivered(self, outbox_id: int) -> None:
        with self.conn:
            self.conn.execute("UPDATE outbox SET status = 'DELIVERED' WHERE id = ?", (outbox_id,))
