"""Operational metrics, latency monitoring, and telemetry collectors for Rex Orchestration (B15).

Provides in-process metric aggregation for case outcomes, escalation rates,
tool failure/timeout rates, and percentile latency profiling.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional


def compute_percentiles(values: List[float]) -> Dict[str, float]:
    """Compute p50, p95, and p99 percentiles from a list of numerical samples."""
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "avg": 0.0}

    s = sorted(values)
    n = len(s)

    def p(pct: float) -> float:
        idx = max(0, min(n - 1, math.ceil(pct * n) - 1))
        return round(float(s[idx]), 2)

    return {
        "p50": p(0.50),
        "p95": p(0.95),
        "p99": p(0.99),
        "max": round(float(s[-1]), 2),
        "avg": round(float(sum(s) / n), 2),
    }


class OrchMetrics:
    """In-memory telemetry and operations metric collector for Rex Orchestration."""

    def __init__(self) -> None:
        self.cases_total: int = 0
        self.outcome_counts: Dict[str, int] = defaultdict(int)
        self.escalation_reasons: Dict[str, int] = defaultdict(int)

        self.tool_calls_total: Dict[str, int] = defaultdict(int)
        self.tool_statuses: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.tool_latencies: Dict[str, List[float]] = defaultdict(list)

        self.case_latencies: List[float] = []
        self.forward_latencies: List[float] = []
        self.tool_round_latencies: List[float] = []

        self.truncation_counts: int = 0
        self.evidence_rounds_total: int = 0

    def record_case_outcome(
        self,
        outcome: str,
        reason: Optional[str] = None,
        duration_ms: float = 0.0,
    ) -> None:
        self.cases_total += 1
        self.outcome_counts[outcome] += 1
        if reason:
            self.escalation_reasons[reason] += 1
        if duration_ms > 0:
            self.case_latencies.append(duration_ms)

    def record_tool_call(self, tool: str, status: str, duration_ms: float) -> None:
        self.tool_calls_total[tool] += 1
        self.tool_statuses[tool][status] += 1
        self.tool_latencies[tool].append(duration_ms)

    def record_evidence_packing(self, is_truncated: bool) -> None:
        self.evidence_rounds_total += 1
        if is_truncated:
            self.truncation_counts += 1

    def record_round_breakdown(self, forward_ms: float, tool_ms: float) -> None:
        self.forward_latencies.append(forward_ms)
        self.tool_round_latencies.append(tool_ms)

    def get_escalation_rate(self) -> float:
        if self.cases_total == 0:
            return 0.0
        escalations = self.outcome_counts.get("ESCALATE", 0)
        return round(escalations / self.cases_total, 4)

    def get_tool_error_rate(self, tool: str) -> float:
        total = self.tool_calls_total.get(tool, 0)
        if total == 0:
            return 0.0
        errors = self.tool_statuses[tool].get("ERROR", 0) + self.tool_statuses[tool].get("TIMEOUT", 0)
        return round(errors / total, 4)

    def snapshot(self) -> Dict[str, Any]:
        """Produce structured metrics snapshot."""
        tool_metrics: Dict[str, Any] = {}
        for t, total in self.tool_calls_total.items():
            tool_metrics[t] = {
                "total_calls": total,
                "statuses": dict(self.tool_statuses[t]),
                "error_rate": self.get_tool_error_rate(t),
                "latencies": compute_percentiles(self.tool_latencies[t]),
            }

        truncation_rate = (
            round(self.truncation_counts / self.evidence_rounds_total, 4)
            if self.evidence_rounds_total > 0
            else 0.0
        )

        return {
            "cases_total": self.cases_total,
            "escalation_rate": self.get_escalation_rate(),
            "outcomes": dict(self.outcome_counts),
            "escalation_reasons": dict(self.escalation_reasons),
            "truncation_rate": truncation_rate,
            "latency_case_ms": compute_percentiles(self.case_latencies),
            "latency_forward_ms": compute_percentiles(self.forward_latencies),
            "latency_tool_round_ms": compute_percentiles(self.tool_round_latencies),
            "tools": tool_metrics,
        }
